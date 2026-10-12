import copy
import json
import os
import time
import uuid
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.configs import GeneratingArguments
from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.configs.base_config import RouterArguments
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.opsd_metrics import response_repetition_metrics
from roll.utils.opsd_prompt import strip_rubric_metadata
from roll.utils.relay_only import apply_rubric_rlsd
from roll.utils.grpo_opsd import (distillation_skipped, fkl_plateau, grpo_opsd_prediction_mask,
                                  load_fkl_history_seed, scheduled_distill_coef)
from roll.pipeline.rlvr.utils import dump_rollout_to_specific_path
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    RunningMoments,
    clusters_have_disjoint_devices,
    agg_loss,
    align_opsd_teacher_log_probs,
    build_domain_routing_context,
    build_opsd_teacher_batch,
    compute_advantage,
    compute_ref_log_probs_with_routing,
    compute_token_reward,
    get_sample_level_mask,
    reduce_metrics,
    reward_postprocess,
    batch_balance,
)
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.offload_states import OffloadStateType
from roll.utils.telemetry import get_tracer, inject_trace_context


logger = get_logger()


def is_lora_training(pipeline_config: RLVRConfig) -> bool:
    return pipeline_config.actor_train.model_args.lora_target is not None


def preprocess_dataset(dataset, prompt_len, encode_function, data_args, teacher_prompt_len=None):
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    encoded_rows = len(dataset)
    dataset = dataset.filter(
        lambda data_i: 5 < len(data_i["input_ids"]) <= prompt_len,
        num_proc=data_args.preprocessing_num_workers,
        desc="Filtering student prompt length",
    )
    student_filtered_rows = encoded_rows - len(dataset)
    teacher_filtered_rows = 0
    if teacher_prompt_len is not None:
        student_retained_rows = len(dataset)
        dataset = dataset.filter(
            lambda data_i: len(data_i["opsd_teacher_prompt_ids"]) <= teacher_prompt_len,
            num_proc=data_args.preprocessing_num_workers,
            desc="Filtering teacher prompt length",
        )
        teacher_filtered_rows = student_retained_rows - len(dataset)
    logger.info(
        f"Encoded {encoded_rows} rows; filtered {student_filtered_rows} by student prompt length "
        f"and {teacher_filtered_rows} by teacher prompt length; retained {len(dataset)} rows."
    )
    return dataset


def get_encode_function(
    template_name,
    tokenizer,
    data_args,
    tag_to_template=None,
    pipeline_config=None,
    include_opsd_teacher=True,
):
    template_funcs = {template_name: get_chat_template(template_name, tokenizer)}
    if tag_to_template:
        for tmpl_name in set(tag_to_template.values()):
            if tmpl_name not in template_funcs:
                template_funcs[tmpl_name] = get_chat_template(tmpl_name, tokenizer)

    tag_key = getattr(data_args, "tag", "tag")
    opsd_enabled = bool(pipeline_config and pipeline_config.use_opsd)
    include_opsd_teacher = opsd_enabled and include_opsd_teacher
    student_thinking = pipeline_config.opsd_student_thinking if opsd_enabled else None
    teacher_thinking = pipeline_config.opsd_teacher_thinking if opsd_enabled else None
    use_privileged_context = pipeline_config.opsd_use_privileged_context if opsd_enabled else False
    privileged_context_key = pipeline_config.opsd_privileged_context_key if opsd_enabled else None
    teacher_prompt_template = pipeline_config.opsd_teacher_prompt_template if opsd_enabled else None

    def render_teacher_text(messages, template_func, privileged_context):
        plain_criteria = getattr(pipeline_config, "opsd_teacher_plain_criteria", False)
        if plain_criteria or getattr(pipeline_config, "opsd_strip_rubric_metadata", False):
            privileged_context = strip_rubric_metadata(privileged_context, keep_polarity=not plain_criteria)
        teacher_messages = list(messages)
        user_idx = next(
            (index for index in range(len(teacher_messages) - 1, -1, -1) if teacher_messages[index]["role"] == "user"),
            None,
        )
        if user_idx is None:
            raise ValueError("OPSD messages must contain at least one user message.")
        student_content = teacher_messages[user_idx]["content"]
        if not isinstance(student_content, str):
            raise ValueError("OPSD currently supports only text user messages.")
        teacher_messages[user_idx] = dict(teacher_messages[user_idx])
        teacher_messages[user_idx]["content"] = teacher_prompt_template.format(
            student_content=student_content,
            privileged_context=privileged_context,
        )
        return template_func(teacher_messages, enable_thinking=teacher_thinking)

    def encode_function(data_i):
        text_list = []
        teacher_text_list = []
        privileged_contexts = None
        if include_opsd_teacher and use_privileged_context:
            if privileged_context_key not in data_i:
                raise ValueError(
                    f"OPSD dataset is missing privileged context column {privileged_context_key!r}."
                )
            privileged_contexts = data_i[privileged_context_key]

        if (message_key := getattr(data_args, "messages", "messages")) in data_i:
            messages_list = data_i[message_key]
            tags = data_i.get(tag_key, [None] * len(messages_list))
            for index, (messages, tag) in enumerate(zip(messages_list, tags)):
                if isinstance(messages, str):
                    messages = json.loads(messages)
                tmpl_name = tag_to_template.get(tag, template_name) if tag_to_template else template_name
                template_func = template_funcs[tmpl_name]
                if opsd_enabled:
                    text_list.append(
                        template_func(messages, enable_thinking=student_thinking)
                    )
                else:
                    text_list.append(template_func(messages))
                if include_opsd_teacher:
                    privileged_context = privileged_contexts[index] if use_privileged_context else ""
                    teacher_text_list.append(
                        render_teacher_text(messages, template_func, privileged_context)
                    )
        elif (prompt_key := getattr(data_args, "prompt", "prompt")) in data_i:
            prompts = data_i[prompt_key]
            tags = data_i.get(tag_key, [None] * len(prompts))
            for index, (prompt, tag) in enumerate(zip(prompts, tags)):
                if opsd_enabled:
                    tmpl_name = tag_to_template.get(tag, template_name) if tag_to_template else template_name
                    template_func = template_funcs[tmpl_name]
                    messages = [{"role": "user", "content": prompt}]
                    text_list.append(
                        template_func(messages, enable_thinking=student_thinking)
                    )
                    if include_opsd_teacher:
                        privileged_context = privileged_contexts[index] if use_privileged_context else ""
                        teacher_text_list.append(
                            render_teacher_text(messages, template_func, privileged_context)
                        )
                else:
                    text_list.append(prompt)
        else:
            raise ValueError("Dataset must contain the configured messages or prompt column.")

        encodings = tokenizer(text_list)
        if include_opsd_teacher:
            teacher_encodings = tokenizer(teacher_text_list)
            encodings["opsd_teacher_prompt_ids"] = teacher_encodings["input_ids"]
        return encodings

    return encode_function


def update_dataset_domain(tag_2_domain: Dict[str, set[str]], row):
    if "domain" in row and row["domain"] is not None:
        return row
    row["domain"] = tag_2_domain.get(row["tag"], "math_rule")
    return row



class RLVRPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.use_ref_model = self.pipeline_config.enable_reference and (not is_lora_training(self.pipeline_config))
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        dataset_paths = []
        if self.pipeline_config.actor_train.data_args.file_name:
            dataset_paths.extend(self.pipeline_config.actor_train.data_args.file_name)

        print(f"load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}")
        dataset = get_dataset(self.pipeline_config.actor_train.data_args)

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            self.val_dataset = get_dataset(self.pipeline_config.validation.data_args)

        # 加上format，然后转ids的func
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(
            template_name,
            self.tokenizer,
            self.pipeline_config.actor_train.data_args,
            tag_to_template=self.pipeline_config.tag_to_template,
            pipeline_config=self.pipeline_config,
        )

        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            data_args=self.pipeline_config.actor_train.data_args,
            teacher_prompt_len=(
                self.pipeline_config.opsd_teacher_max_prompt_length or self.pipeline_config.prompt_length
                if self.pipeline_config.use_opsd
                else None
            ),
        )
        # update domain field
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )
        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        if self.val_dataset:
            validation_data_args = self.pipeline_config.validation.data_args
            validation_template_name = (
                self.pipeline_config.global_template
                if self.pipeline_config.global_template
                else validation_data_args.template
            )
            val_encode_function = get_encode_function(
                validation_template_name,
                self.tokenizer,
                validation_data_args,
                tag_to_template=self.pipeline_config.tag_to_template,
                pipeline_config=self.pipeline_config,
                include_opsd_teacher=False,
            )
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.val_prompt_length,
                val_encode_function,
                data_args=validation_data_args,
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=validation_data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )
            assert "domain" in self.val_dataset.column_names, "domain field should set in val dataset"

        assert "domain" in dataset.column_names, "domain field should set in dataset"
        print(dataset)

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )
        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        # Build cluster creation specs (label, name, worker_cls, worker_config)
        # All clusters are created in parallel since Cluster.__init__ only allocates
        # placement groups and creates Ray actors (no GPU work), which is thread-safe.
        cluster_specs: List[tuple] = []

        cluster_specs.append((
            "actor_train",
            self.pipeline_config.actor_train.name,
            self.pipeline_config.actor_train.worker_cls,
            self.pipeline_config.actor_train,
        ))
        cluster_specs.append((
            "actor_infer",
            self.pipeline_config.actor_infer.name,
            self.pipeline_config.actor_infer.worker_cls,
            self.pipeline_config.actor_infer,
        ))

        # use unwrapped model as reference for lora training
        if self.use_ref_model:
            for name, ref_cfg in self.pipeline_config.reference_configs.items():
                cluster_specs.append((
                    f"reference_{name}",
                    ref_cfg.name,
                    ref_cfg.worker_cls,
                    ref_cfg,
                ))

        if self.pipeline_config.use_rubric_rlsd and self.pipeline_config.rubric_rlsd_teacher is not None:
            teacher_cfg = self.pipeline_config.rubric_rlsd_teacher
            cluster_specs.append(("rubric_rlsd_teacher", teacher_cfg.name, teacher_cfg.worker_cls, teacher_cfg))

        if self.pipeline_config.use_grpo_opsd and self.pipeline_config.opsd_teacher_refresh_step is not None:
            # A separate privileged teacher, so the refresh never touches the KL reference.
            teacher_cfg = copy.deepcopy(next(iter(self.pipeline_config.reference_configs.values())))
            teacher_cfg.name = "grpo_opsd_teacher"
            cluster_specs.append(("grpo_opsd_teacher", teacher_cfg.name, teacher_cfg.worker_cls, teacher_cfg))

        if self.pipeline_config.adv_estimator == "gae":
            cluster_specs.append((
                "critic",
                self.pipeline_config.critic.name,
                self.pipeline_config.critic.worker_cls,
                self.pipeline_config.critic,
            ))

        for key, worker_config in self.pipeline_config.rewards.items():
            cluster_specs.append((
                f"reward_{key}",
                f"reward-{key}",
                worker_config.worker_cls,
                worker_config,
            ))

        # Create reward model cluster (shared InferWorker + vLLM for LLM-as-judge)
        self.reward_model_cluster = None
        self.reward_model_scheduler = None
        has_reward_model = (
            self.pipeline_config.reward_model is not None
            and self.pipeline_config.reward_model.device_mapping
            and len(self.pipeline_config.reward_model.device_mapping) > 0
        )
        if has_reward_model:
            cluster_specs.append((
                "reward_model",
                self.pipeline_config.reward_model.name,
                self.pipeline_config.reward_model.worker_cls,
                self.pipeline_config.reward_model,
            ))

        # Create all clusters in parallel
        clusters = self.create_clusters_parallel(cluster_specs)

        # Assign clusters to attributes (preserving original order semantics)
        self.actor_train = clusters["actor_train"]
        self.actor_infer = clusters["actor_infer"]
        download_clusters = [self.actor_train, self.actor_infer]
        self.rubric_rlsd_teacher = clusters.get("rubric_rlsd_teacher")
        if self.rubric_rlsd_teacher is not None:
            download_clusters.append(self.rubric_rlsd_teacher)
        self.grpo_opsd_teacher = clusters.get("grpo_opsd_teacher")
        if self.grpo_opsd_teacher is not None:
            download_clusters.append(self.grpo_opsd_teacher)

        if self.use_ref_model:
            self.references: Dict[str, Any] = {}
            for name in self.pipeline_config.reference_configs.keys():
                self.references[name] = clusters[f"reference_{name}"]
            download_clusters.extend(self.references.values())
            # Backward compat: self.reference points to the first teacher
            self.reference = self.references[list(self.references.keys())[0]]

        if self.pipeline_config.adv_estimator == "gae":
            self.critic = clusters["critic"]
            download_clusters.append(self.critic)

        self.rewards: Dict[str, Any] = {
            key: clusters[f"reward_{key}"] for key in self.pipeline_config.rewards.keys()
        }
        download_clusters.extend(self.rewards.values())

        if has_reward_model:
            self.reward_model_cluster = clusters["reward_model"]
            download_clusters.append(self.reward_model_cluster)

        self.download_models(*download_clusters)

        # Create RouterManager for reward model cluster (Ray named actor)
        if self.reward_model_cluster:
            self.reward_model_scheduler = ray.remote(RouterManager).options(
                name=f"RewardModelScheduler-{self.pipeline_config.reward_model.name}",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                actor_cluster=self.reward_model_cluster,
                router_args=RouterArguments(router_name="PromptAffinityRouter"),
                num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
            )
            ray.get(self.reward_model_scheduler.initialize.remote())
            logger.info(
                f"Created reward model scheduler: RewardModelScheduler-{self.pipeline_config.reward_model.name}"
            )

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(
                pipeline_config=self.pipeline_config,
                actor_cluster=self.actor_infer,
                reward_clusters={domain: self.rewards[domain]},
                dataset=self.domain_datasets[domain],
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                state=self.state.kv.get(f"scheduler_state_{domain}", None),
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False
            self.val_generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(
                pipeline_config=val_pipeline_config,
                actor_cluster=self.actor_infer,
                reward_clusters=self.rewards,
                dataset=self.val_dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs=dict(max_length=self.pipeline_config.val_prompt_length, padding="max_length"),
                is_val=True,
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.reward_model_cluster:
            refs.extend(self.reward_model_cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if self.use_ref_model:
            if clusters_have_disjoint_devices(self.references):
                ref_init_refs = []
                for ref_cluster in self.references.values():
                    ref_init_refs.extend(ref_cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
                ray.get(ref_init_refs)
            else:
                for ref_cluster in self.references.values():
                    ref_cluster.initialize(pipeline_config=self.pipeline_config, blocking=True)

        if self.rubric_rlsd_teacher is not None:
            self.rubric_rlsd_teacher.initialize(pipeline_config=self.pipeline_config, blocking=True)
        if self.grpo_opsd_teacher is not None:
            self.grpo_opsd_teacher.initialize(pipeline_config=self.pipeline_config, blocking=True)

        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        ray.get([scheduler.initialize.remote() for scheduler in self.generate_schedulers.values()])
        if self.val_dataset:
            ray.get(self.val_generate_scheduler.initialize.remote())

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        if self.pipeline_config.use_opsd and self.pipeline_config.opsd_teacher_update_mode == "student_sync":
            for reference in self.references.values():
                self.set_model_update_pair(
                    src_cluster=self.actor_train,
                    tgt_cluster=reference,
                    frequency=self.pipeline_config.opsd_teacher_update_frequency,
                )

        self._setup_rubric_rlsd_teacher_updates()
        self._setup_grpo_opsd_teacher_refresh()

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

    def _setup_rubric_rlsd_teacher_updates(self) -> None:
        """Only synchronize the privileged teacher; never update the KL reference."""
        if (self.pipeline_config.use_rubric_rlsd
                and self.pipeline_config.rubric_rlsd_teacher_mode == "student_sync"):
            self.set_model_update_pair(
                src_cluster=self.actor_train,
                tgt_cluster=self.rubric_rlsd_teacher,
                frequency=self.pipeline_config.rubric_rlsd_teacher_update_frequency,
            )

    def _setup_grpo_opsd_teacher_refresh(self) -> None:
        """Prepare the one-time student snapshot into the privileged teacher, never the KL reference."""
        self.grpo_opsd_teacher_refresh = None
        refresh_step = self.pipeline_config.opsd_teacher_refresh_step
        if self.grpo_opsd_teacher is None or refresh_step is None:
            return
        # The refreshed teacher is not checkpointed; a later resume would silently keep the initial teacher.
        if self.state.step > refresh_step:
            raise ValueError(
                f"Teacher refresh after step {refresh_step} cannot resume from step {self.state.step}; "
                f"resume at or before step {refresh_step}."
            )
        self.grpo_opsd_teacher_refresh = ModelUpdateGroup(
            src_cluster=self.actor_train, tgt_cluster=self.grpo_opsd_teacher,
            pipeline_config=self.pipeline_config, frequency=1,
        )

    def _maybe_refresh_grpo_opsd_teacher(self, global_step: int) -> Dict:
        """Copy the student into the privileged teacher at the start of the step after the refresh step."""
        if self.grpo_opsd_teacher_refresh is None:
            return {}
        refresh_step = self.pipeline_config.opsd_teacher_refresh_step
        if global_step != refresh_step + 1:
            return {}
        metrics = self.grpo_opsd_teacher_refresh.model_update(global_step)
        self.grpo_opsd_teacher.process_weights_after_loading()
        logger.info(f"OPSD privileged teacher refreshed from the student after step {refresh_step}.")
        return {**metrics, "opsd_teacher_refresh/after_step": refresh_step}

    def _apply_grpo_opsd_schedule(self, batch: DataProto, global_step: int, metrics_mgr) -> bool:
        """Pass this step's OPSD weight to the actor; return whether the teacher and distillation are skipped."""
        if not self.pipeline_config.use_grpo_opsd:
            return False
        auto_step = self.state.kv.get("opsd_auto_switch_step")
        coef = scheduled_distill_coef(self.pipeline_config, global_step, auto_step)
        skipped = distillation_skipped(self.pipeline_config, global_step, auto_step)
        batch.meta_info["opsd_distill_loss_coef"] = coef
        batch.meta_info["opsd_distill_skipped"] = skipped
        metrics_mgr.add_metrics({"actor/opsd_distill_coef": coef, "actor/opsd_distill_skipped": float(skipped)})
        return skipped

    def _update_opsd_auto_switch(self, global_step: int, metrics_mgr) -> None:
        """Record this step's selected-token FKL; at the plateau, fix the switch after this step (checkpointed)."""
        cfg = self.pipeline_config
        if not (cfg.use_grpo_opsd and cfg.opsd_distill_auto_switch) or "opsd_auto_switch_step" in self.state.kv:
            return
        metrics = metrics_mgr.metrics
        value = metrics.get("actor/opsd_selected_token_fkl@sum", metrics.get("actor/opsd_selected_token_fkl"))
        if value is None:  # no distillation this step
            return
        if metrics.get("actor/opsd_selected_token_fraction@sum", 1.0) == 0:  # nothing distilled: no signal
            return
        if "opsd_auto_switch_history" not in self.state.kv:
            seed = load_fkl_history_seed(getattr(cfg, "opsd_auto_switch_history_seed", None), global_step)
            if seed:
                logger.info(f"OPSD auto switch: seeded {len(seed)} history entries up to step {seed[-1][0]}.")
            self.state.kv["opsd_auto_switch_history"] = seed
        history = self.state.kv["opsd_auto_switch_history"]
        history.append([global_step, float(value)])
        plateau, drop = fkl_plateau(history, cfg.opsd_auto_switch_window, cfg.opsd_auto_switch_rel_tol,
                                    cfg.opsd_auto_switch_patience, cfg.opsd_auto_switch_min_step)
        if drop is not None:
            metrics_mgr.add_metrics({"actor/opsd_auto_switch_fkl_drop": drop})
        if plateau:
            self.state.kv["opsd_auto_switch_step"] = global_step
            metrics_mgr.add_metrics({"actor/opsd_auto_switch_step": global_step})
            logger.info(f"OPSD selected-token FKL plateaued; switching the distillation weight after step {global_step}.")

    def _grpo_opsd_privileged_teachers(self) -> Dict[str, Any]:
        """Clusters scoring privileged context; the original-context KL keeps using self.references."""
        if self.grpo_opsd_teacher is None:
            return self.references
        return {name: self.grpo_opsd_teacher for name in self.references}

    @torch.no_grad()
    def save_metrics(self, batch):
        def remove_leading_zeros(A, r_mask):
            B = []
            for i in range(len(A)):
                row = A[i]
                mask = (r_mask[i] != 0).to(torch.int32)
                if not mask.any():  # 如果该行全为零
                    B.append([])  # 添加空列表
                else:
                    first_non_zero = mask.argmax().item()  # 找到第一个非零元素的索引
                    B.append(row[first_non_zero:].tolist())
            return B

        res_dict = {}
        batch_size = batch.batch["old_log_probs"].shape[0]
        res_dict["log_probs2"] = remove_leading_zeros(
            batch.batch["old_log_probs2"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["logprobs"] = remove_leading_zeros(
            batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
        )  # 这里注意batch.batch["old_log_probs"]是right pad 格式
        res_dict["new_response_mask"] = remove_leading_zeros(
            batch.batch["response_mask"][:, 1:], batch.batch["response_mask"][:, 1:]
        )
        res_dict["prompt"] = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
        res_dict["response"] = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
        res_dict["old_log_probs2_entropy"] = remove_leading_zeros(batch.batch["old_log_probs2_entropy"], batch.batch["response_mask"][:, 1:])
        res_dict["ref_logprobs"] = remove_leading_zeros(
            batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["values"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["token_rewards"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["reward"] = batch.batch["response_level_rewards"].numpy().tolist()
        step = batch.meta_info["global_step"]

        assert (
            len(res_dict["logprobs"]) == batch_size
        ), f"len of logprobs is : {len(res_dict['logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["ref_logprobs"]) == batch_size
        ), f"len of ref_logprobs is : {len(res_dict['ref_logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["log_probs2"]) == batch_size
        ), f"len of log_probs2 is : {len(res_dict['log_probs2'])}, len of batch size is :{batch_size}"

        res = []
        for i in range(batch_size):
            temp_dict = {}
            for k, _ in res_dict.items():
                temp_dict[k] = res_dict[k][i]
            temp_dict["step"] = step
            temp_dict["response_tokens"] = self.tokenizer.convert_ids_to_tokens(
                batch.batch["responses"][i], skip_special_tokens=False
            )
            res.append(temp_dict)

        file_dir = self.pipeline_config.save_logging_board_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 用时间命名domain batch 文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        milliseconds = str(int(time.time() * 1000) % 1000).zfill(3)
        file_name = f"data_{current_time}_{milliseconds}.jsonl"

        try:
            with open(file_dir + file_name, "a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.info(f"Writing files catch error :{e}")

    def get_generation_config(self, generating_args: Optional[GeneratingArguments] = None):
        generating_args = (
            generating_args if generating_args is not None else self.actor_infer.worker_config.generating_args
        )
        generation_config = generating_args.to_dict()
        if self.pipeline_config.async_pipeline:
            generation_config["logprobs"] = 1
        return generation_config

    @torch.no_grad()
    def should_validate(self, global_step: int) -> bool:
        """Select pre-rollout evaluations; subclasses may use completed-update milestones."""
        return bool(self.val_dataset) and global_step % self.pipeline_config.eval_steps == 0

    def prepare_opsd_mask(self, batch: DataProto, prediction_mask: torch.Tensor,
                          global_step: int) -> torch.Tensor:
        """Allow an experiment to select distillation rows without changing GRPO masks."""
        return prediction_mask

    def record_validation_batch(self, batch: DataProto, global_step: int) -> None:
        """Optional experiment audit before evaluation tensors are consumed."""

    def after_train_step(self, global_step: int) -> dict:
        """Optional post-update experiment evaluation, including the final update."""
        return {}

    def validate_final_policy(self) -> None:
        """Evaluate the final saved policy at the completed-rollout count (400 by default)."""
        completed_steps = self.state.step + 1
        if not self.val_dataset or completed_steps != self.pipeline_config.max_steps:
            return
        # The last regular validation precedes its update. Transfer the final
        # actor weights before requesting the additional endpoint validation.
        self.actor_train.offload_states(blocking=True)
        self.model_update(completed_steps)
        self.actor_infer.load_states(blocking=True)
        for reward_cluster in self.rewards.values():
            reward_cluster.load_states()
        try:
            metrics = self.val(global_step=completed_steps)
            metrics.update({"validation/completed_steps": completed_steps,
                            "validation/checkpoint_step": self.state.step})
            self.tracker.log(values=metrics, step=completed_steps)
        finally:
            self.actor_infer.offload_states(blocking=True)
            for reward_cluster in self.rewards.values():
                reward_cluster.offload_states()

    def run(self):
        # 计算tokens per second 系统吞吐

        # 创建一个专门管理监控指标的类
        metrics_mgr = MetricsManager()
        tracer = get_tracer("driver")

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        metrics_mgr.timers["tps"] = tps_timer
        metrics_mgr.timers["actor_infer"] = actor_infer_timer
        metrics_mgr.timers["actor_infer_response"] = actor_infer_response_timer
        metrics_mgr.timers["actor_train"] = actor_train_timer

        pre_step_total_time = 0
        if self.pipeline_config.async_pipeline:
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()
            if self.reward_model_cluster:
                self.reward_model_cluster.load_states()

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with (tps_timer, Timer(name="step_total", logger=None) as step_total_timer,
                tracer.start_as_current_span("pipeline_step", attributes={"global_step": global_step}),
                ExitStack() as defer,
            ):
                # if global_step > self.state.step + 1:
                logger.info(f"pre_step_total_time: {pre_step_total_time}")
                metrics_mgr.add_metric("time/step_total", pre_step_total_time)
                batch: DataProto = DataProto(
                    meta_info={
                        "global_step": global_step,
                        "collect_unfinished": self.pipeline_config.async_pipeline,
                        "reward_system_config": self.pipeline_config.reward_system_config
                        }
                )

                # 先model update，resume时不需要保存infer cluster的状态
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                with Timer(name="step_stop_server", logger=None) as step_stop_server_timer, \
                        tracer.start_as_current_span("stop_server"):
                    if self.pipeline_config.async_pipeline:
                        ray.get([scheduler.pause_sampling.remote() for scheduler in self.generate_schedulers.values()])
                        self.actor_infer.offload_states(include=OffloadStateType.other_params)
                metrics_mgr.add_metric("time/step_stop_server", step_stop_server_timer.last)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer, \
                        tracer.start_as_current_span("model_update"):
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    metrics_mgr.add_metrics(self._maybe_refresh_grpo_opsd_teacher(global_step))
                    batch.meta_info["generation_config"] = self.get_generation_config()
                metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                self.actor_infer.load_states(blocking=True)
                if not self.pipeline_config.async_pipeline:
                    for reward_cluster in self.rewards.values():
                        reward_cluster.load_states()

                if self.should_validate(global_step):
                    with Timer(name="val_step", logger=None) as val_step_timer, \
                            tracer.start_as_current_span("validation"):
                        val_metrics = self.val(global_step=global_step)
                    metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                # 要按domain group by生成对应的batch
                with (
                    actor_infer_timer,
                    actor_infer_response_timer,
                    Timer(name="step_generate", logger=None) as step_generate_timer,
                    tracer.start_as_current_span("generate"),
                ):
                    domain_batches = {}
                    scheduler_refs = {}
                    for domain, scheduler in self.generate_schedulers.items():
                        inject_trace_context(batch.meta_info)
                        scheduler_refs[domain] = scheduler.get_batch.remote(
                            data=batch, global_step=global_step, batch_size=self.domain_batch_size[domain]
                        )
                    for domain, scheduler_ref in scheduler_refs.items():
                        domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        )
                        domain_batches[domain] = domain_batch
                    generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                    dump_rollout_to_specific_path(self.pipeline_config.rollout_dump_dir, global_step, generate_output, self.tokenizer)
                    generate_output.meta_info.pop("is_offload_states", None)

                    if not self.pipeline_config.async_pipeline:
                        ray.get([scheduler.pause_sampling.remote() for scheduler in self.generate_schedulers.values()])
                        self.actor_infer.offload_states()
                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        if self.reward_model_cluster:
                            self.reward_model_cluster.offload_states()
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output
                if self.pipeline_config.use_ruscarl:
                    counts = batch.non_tensor_batch["ruscarl_criteria_count"].astype(float)
                    totals = batch.non_tensor_batch["ruscarl_total_criteria"].astype(float)
                    metrics_mgr.add_metrics({
                        "ruscarl/ratio_mean": float(batch.non_tensor_batch["ruscarl_ratio"].astype(float).mean()),
                        "ruscarl/criteria_mean": float(counts.mean()),
                        "ruscarl/revealed_fraction": float((counts / np.maximum(totals, 1)).mean()),
                        "ruscarl/scaffolded_fraction": float((counts > 0).mean()),
                    })
                if self.pipeline_config.use_opsd or self.pipeline_config.use_cripo:
                    metrics_mgr.add_metrics(response_repetition_metrics(
                        batch.batch["input_ids"], batch.batch["response_mask"], self.tokenizer.all_special_ids
                    ))
                defer.callback(lambda b=batch: DataProto.drop(b))
                batch.meta_info["global_step"] = global_step
                batch.meta_info["_broadcast_non_tensor_batch"] = True
                batch.meta_info["loss_mask_keys"] = ['response_mask', 'final_response_mask']
                skip_distill = self._apply_grpo_opsd_schedule(batch, global_step, metrics_mgr)
                batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)

                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer, \
                        tracer.start_as_current_span("ref_log_probs"):
                    if self.pipeline_config.enable_reference:
                        if not self.use_ref_model:
                            # LoRA branch: use actor_train with disabled adapter
                            worker_config = self.pipeline_config.actor_train
                            batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                            batch.meta_info["disable_adapter"] = True
                            batch.meta_info["is_offload_states"] = False
                            if worker_config.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    worker_config.max_tokens_per_microbatch_in_infer,
                                    worker_config.sequence_length_round_in_infer,
                                    worker_config.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    worker_config.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "reference/compute_log_probs",
                                )
                                metrics_mgr.add_metrics(dynamic_batching_metrics)
                            ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                            metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        else:
                            saved_routed_experts = batch.batch.pop("routed_experts", None)
                            domain_to_teacher_names, domain_values = build_domain_routing_context(
                                batch, self.pipeline_config, domain_key="domain"
                            )
                            # Fixed OPSD teacher has the same initial weights as the KL reference.
                            # Score original context separately; privileged logprobs must never
                            # become the GRPO reference regularizer.
                            if self.pipeline_config.use_grpo_opsd:
                                batch = compute_ref_log_probs_with_routing(
                                    batch=batch, references=self.references,
                                    pipeline_config=self.pipeline_config,
                                    domain_to_teacher_names=domain_to_teacher_names,
                                    domain_values=domain_values,
                                    metrics_fn=metrics_mgr.add_metrics,
                                )
                                batch.batch["grpo_reference_log_probs"] = batch.batch["ref_log_probs"].clone()
                            reference_batch_builder = None
                            if self.pipeline_config.use_opsd:
                                feedback = self.pipeline_config.opsd_feedback_mode != "none"
                                if feedback:
                                    batch.non_tensor_batch['opsd_teacher_prompt_ids'] = batch.non_tensor_batch.pop(
                                        'opsd_feedback_teacher_prompt_ids')
                                reference_batch_builder = partial(
                                    build_opsd_teacher_batch,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    teacher_sequence_length=(self.pipeline_config.opsd_feedback_sequence_length if feedback else None),
                                    opsd_logits_topk=(
                                        self.pipeline_config.opsd_logits_topk
                                        if self.pipeline_config.uses_opsd_distribution_loss
                                        else 0
                                    ),
                                )
                            # After a scheduled switch to zero weight only the original-context KL remains.
                            if not skip_distill:
                                batch = compute_ref_log_probs_with_routing(
                                    batch=batch,
                                    references=self._grpo_opsd_privileged_teachers(),
                                    pipeline_config=self.pipeline_config,
                                    domain_to_teacher_names=domain_to_teacher_names,
                                    domain_values=domain_values,
                                    metrics_fn=metrics_mgr.add_metrics,
                                    reference_batch_builder=reference_batch_builder,
                                    reference_result_aligner=(
                                        align_opsd_teacher_log_probs if self.pipeline_config.use_opsd else None
                                    ),
                                )
                            if self.pipeline_config.use_grpo_opsd:
                                batch.batch["ref_log_probs"] = batch.batch.pop("grpo_reference_log_probs")
                            if self.pipeline_config.use_opsd:
                                teacher_lengths = np.asarray([
                                    len(ids) for ids in batch.non_tensor_batch["opsd_teacher_prompt_ids"]
                                ])
                                metrics_mgr.add_metrics({
                                    "token/teacher_prompt_length/mean": float(teacher_lengths.mean()),
                                    "token/teacher_prompt_length/max": int(teacher_lengths.max()),
                                    "token/teacher_prompt_length/min": int(teacher_lengths.min()),
                                })
                                batch.non_tensor_batch.pop("opsd_teacher_prompt_ids", None)
                            if saved_routed_experts is not None:
                                batch.batch["routed_experts"] = saved_routed_experts
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer, \
                        tracer.start_as_current_span("old_log_probs_and_values"):
                    if self.pipeline_config.enable_reference and not self.use_ref_model:
                        batch.meta_info["disable_adapter"] = False
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                        if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                            batch, dynamic_batching_metrics = dynamic_batching_shard(
                                batch,
                                self.actor_train.dp_size,
                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                "actor_train/compute_log_probs",
                            )
                            metrics_mgr.add_metrics(dynamic_batching_metrics)
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)

                        # Customize_logging metrics, Double check call twice
                        if self.pipeline_config.save_logging_board_dir:
                            old_log_probs_refs2: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                batch, blocking=False
                            )
                            old_log_probs2 = DataProto.materialize_concat(data_refs=old_log_probs_refs2)
                            batch.batch["old_log_probs2"] = old_log_probs2.batch["log_probs"]
                            batch.batch["old_log_probs2_entropy"] = old_log_probs2.batch["entropy"]

                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy

                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                    else:
                        # Use zeros when optimization is enabled
                        batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))

                    # Mock ref_log_probs using old_log_probs if reference is disabled
                    if not self.pipeline_config.enable_reference:
                        batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                        # Multi-teacher: also mock per-teacher keys (compute_advantage reads them)
                        for name in self.pipeline_config.reference_configs.keys():
                            batch.batch[f"ref_log_probs_{name}"] = batch.batch["old_log_probs"].clone()
                            batch.batch[f"ref_log_probs_{name}_mask"] = torch.ones(
                                batch.batch.batch_size[0], dtype=torch.bool, device=batch.batch["attention_mask"].device
                            )
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                # 要按domain group by处理reward
                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. 处理mask相关策略， 获取sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer, \
                            tracer.start_as_current_span("get_sample_level_mask"):
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_sample_level_mask": get_sample_level_mask_timer.last})

                    # 2. 处理reward相关策略
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer, \
                            tracer.start_as_current_span("reward_postprocess"):
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})

                    # 3. 计算token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer, \
                            tracer.start_as_current_span("compute_token_reward"):
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})

                    # 4. 计算advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer, \
                            tracer.start_as_current_span("compute_advantage"):
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                            pipeline_config=self.pipeline_config,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        if self.pipeline_config.use_cripo:
                            from roll.utils.relay_only import prepare_cripo
                            metrics_mgr.add_domain_metrics(domain, prepare_cripo(
                                domain_batch, self.tokenizer, self.pipeline_config,
                            ))
                        batch_list.append(domain_batch)
                    metrics_mgr.add_domain_metrics(domain, {"time/compute_advantage": compute_advantage_timer.last})
                    if (
                        self.pipeline_config.save_logging_board_dir
                        and not self.pipeline_config.uses_opsd_direct_loss
                    ):
                        self.save_metrics(domain_batch)

                batch = DataProto.concat(batch_list)
                if self.pipeline_config.use_ruscarl:
                    from roll.utils.relay_only import scaffold_group_metrics
                    # BaseConfig consumes response_length during initialization.
                    metrics_mgr.add_metrics(scaffold_group_metrics(
                        batch, self.pipeline_config.sequence_length - self.pipeline_config.prompt_length,
                    ))
                if self.pipeline_config.use_rubric_rlsd:
                    with Timer(name="rubric_rlsd", logger=None) as rlsd_timer:
                        metrics_mgr.add_metrics(apply_rubric_rlsd(
                            batch, self.actor_train, self.tokenizer, self.pipeline_config,
                            teacher_actor=self.rubric_rlsd_teacher,
                        ))
                    metrics_mgr.add_metric("time/rubric_rlsd", rlsd_timer.last)
                if batch.batch["final_response_mask"].sum() == 0:
                    logger.info("Warning: final_response_mask.sum() == 0! Current step will be skipped.")
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 1)
                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())
                    self.do_checkpoint(global_step=global_step)
                    self.tracker.log(values=metrics, step=global_step)
                    continue
                else:
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 0)

                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                if not (self.pipeline_config.use_cripo or self.pipeline_config.grpo_ppo_mini_batch_size):
                    batch.pop("prompt_id")

                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                if self.pipeline_config.enable_old_logprobs_recompute:
                    batch, corr_metrics = apply_train_infer_correction_to_batch(self.pipeline_config, batch,
                                                                                update_mask_keys=batch.meta_info[
                                                                                    'loss_mask_keys'])
                    metrics_mgr.add_metrics(corr_metrics)

                response_lengths = batch.batch["response_mask"].sum(-1)
                metrics_mgr.add_metrics({
                    "token/response_truncation_rate": (
                        response_lengths >= self.pipeline_config.sequence_length - self.pipeline_config.prompt_length
                    ).float().mean().item(),
                })

                if self.pipeline_config.uses_opsd_distribution_loss and not skip_distill:
                    prediction_mask = torch.zeros_like(batch.batch["response_mask"], dtype=torch.bool)
                    prediction_mask[:, :-1] = batch.batch["final_response_mask"].bool()
                    if self.pipeline_config.use_grpo_opsd:
                        prediction_mask = grpo_opsd_prediction_mask(
                            batch.batch["advantages"], batch.batch["final_response_mask"],
                            self.pipeline_config.opsd_trajectory_selection,
                            unsatisfied_positive_count=batch.batch.get("rubric_unsatisfied_positive_count"),
                            zero_min_unsatisfied_positive=self.pipeline_config.opsd_zero_min_unsatisfied_positive,
                            rubric_scores=batch.batch.get("scores"),
                            score_exclusive_max=self.pipeline_config.opsd_score_exclusive_max,
                        )
                        batch.meta_info["loss_mask_keys"].append("opsd_prediction_mask")
                    else:
                        batch.meta_info["loss_mask_keys"] = ["opsd_prediction_mask"]
                    if self.pipeline_config.opsd_score_weighting == "positive_lowzero_plateau":
                        from roll.utils.grpo_opsd import adaptive_score_parameters, plateau_gaussian_opsd_weights

                        center = self.pipeline_config.opsd_score_gaussian_center
                        sigma = self.pipeline_config.opsd_score_gaussian_sigma
                        if self.pipeline_config.opsd_score_adaptive:
                            cfg = self.pipeline_config
                            valid = batch.batch["final_response_mask"].bool()
                            judge_ok = batch.batch.get("rubric_parse_ok")
                            if judge_ok is None:
                                raise ValueError("Adaptive OPSD requires per-response judge flags.")
                            eligible = (valid.any(-1) & (judge_ok == 1)
                                        & (response_lengths < cfg.sequence_length - cfg.prompt_length)
                                        & torch.isfinite(batch.batch["advantages"].masked_fill(~valid, 0)).all(-1))
                            center, sigma, score_state, score_metrics = adaptive_score_parameters(
                                batch.batch["scores"], eligible, self.state.kv.get("opsd_score_ema"),
                                cfg.opsd_score_quantile_low, cfg.opsd_score_quantile_high,
                                cfg.opsd_score_ema_rate, cfg.opsd_score_sigma_min, center, sigma,
                            )
                            if score_state is not None:
                                self.state.kv["opsd_score_ema"] = score_state
                            metrics_mgr.add_metrics(score_metrics)
                        weights = plateau_gaussian_opsd_weights(
                            batch.batch["scores"], batch.batch["advantages"], batch.batch["final_response_mask"],
                            batch.batch.get("rubric_parse_ok"), response_lengths,
                            self.pipeline_config.sequence_length - self.pipeline_config.prompt_length,
                            center, sigma,
                            self.pipeline_config.opsd_score_gaussian_floor,
                            include_zero=self.pipeline_config.opsd_plateau_include_zero,
                            zero_score_gate=self.pipeline_config.opsd_plateau_zero_score_gate,
                        )
                        from roll.utils.grpo_opsd import ablate_score_weights

                        weights = ablate_score_weights(
                            weights, batch.batch["final_response_mask"], self.pipeline_config.opsd_score_ablation,
                        )
                        if self.pipeline_config.opsd_negative_weight:
                            # Flip advantage signs to reuse the exact quality and zero-tolerance gates.
                            negative_mask = plateau_gaussian_opsd_weights(
                                batch.batch["scores"], -batch.batch["advantages"],
                                batch.batch["final_response_mask"], batch.batch.get("rubric_parse_ok"),
                                response_lengths,
                                self.pipeline_config.sequence_length - self.pipeline_config.prompt_length,
                                center, sigma, 1.0, include_zero=False,
                            )
                            weights = weights + self.pipeline_config.opsd_negative_weight * negative_mask
                        normalization_mask = torch.zeros_like(prediction_mask)
                        normalization_mask[:, :-1] = batch.batch["final_response_mask"].bool()
                        batch.batch["opsd_normalization_mask"] = normalization_mask
                        batch.batch["opsd_trajectory_weights"] = weights
                        batch.meta_info["loss_mask_keys"].append("opsd_normalization_mask")
                        prediction_mask = normalization_mask & (weights > 0).unsqueeze(-1)
                    prediction_mask = self.prepare_opsd_mask(batch, prediction_mask, global_step)
                    batch.batch["opsd_prediction_mask"] = prediction_mask
                    response_tokens = batch.batch["response_mask"].sum().item()
                    valid_tokens = prediction_mask.sum().item()
                    metrics_mgr.add_metrics({
                        "actor/opsd_valid_tokens": valid_tokens,
                        "actor/opsd_valid_token_fraction": valid_tokens / max(response_tokens, 1),
                        "actor/opsd_valid_samples": prediction_mask.any(-1).sum().item(),
                    })

                with Timer(name="step_train", logger=None) as step_train_timer, \
                        tracer.start_as_current_span("train"):
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # Reorder data for DP rank load balancing
                            if self.pipeline_config.grpo_ppo_mini_batch_size:
                                from roll.utils.rubric_training import balance_group_batch
                                batch_balance_metrics = balance_group_batch(
                                    batch, self.pipeline_config, self.actor_train.dp_size)
                            elif self.pipeline_config.use_cripo:
                                from roll.utils.relay_only import balance_cripo_batch
                                batch_balance_metrics = balance_cripo_batch(
                                    batch, self.pipeline_config, self.actor_train.dp_size)
                            else:
                                batch_balance_metrics = batch_balance(batch, dp_size=self.actor_train.dp_size,
                                    minibatch_size=self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                                    * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps
                                    * self.actor_train.dp_size, logging_prefix="global_seqlen/actor_train")
                            metrics_mgr.add_metrics(batch_balance_metrics)
                            # update actor
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "actor_train/train_step",
                                )
                                metrics_mgr.add_metrics(dynamic_batching_metrics)
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))
                            self._update_opsd_auto_switch(global_step, metrics_mgr)

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                metrics_mgr.add_metric("time/step_train", step_train_timer.last)
                metrics_mgr.add_metrics(self.after_train_step(global_step))

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
            pre_step_total_time = step_total_timer.last

        self.validate_final_policy()
        ray.get([scheduler.shutdown.remote() for scheduler in self.generate_schedulers.values()])
        if self.val_dataset:
            ray.get(self.val_generate_scheduler.shutdown.remote())

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self, global_step):
        defer = ExitStack()
        val_metrics_mgr = MetricsManager()
        tracer = get_tracer("driver")
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer, \
                tracer.start_as_current_span("val_generate"):
            inject_trace_context(batch.meta_info)
            reward_system_config = copy.deepcopy(self.pipeline_config.reward_system_config)
            experiment_id = reward_system_config.get("experiment_id", 'roll_default_task_id')
            # TODO: 当前通过在 experiment_id 后加上_val 的方式来区分 reward system train 和 val 的日志记录, 后续考虑更优雅的 train 和 val 区分方式。
            reward_system_config["experiment_id"] = experiment_id + '_val'
            batch.meta_info.update({
                "is_offload_states": False,
                "generation_config": self.pipeline_config.validation.generating_args.to_dict(),
                "global_step": global_step,
                "reward_system_config": reward_system_config,
            })

            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, global_step=global_step, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )

            generate_output.meta_info.pop("is_offload_states", None)

        # Read the completed timer and keep evaluation separate from rollout time.
        val_metrics_mgr.add_metric("val/time/generation_and_judging", step_generate_timer.last)

        batch = generate_output
        self.record_validation_batch(batch, global_step)
        if self.pipeline_config.use_opsd:
            val_metrics_mgr.add_metrics({f"val/{k}": v for k, v in response_repetition_metrics(
                batch.batch["input_ids"], batch.batch["response_mask"], self.tokenizer.all_special_ids
            ).items()})
            for tag, group in batch.group_by("tag").items():
                val_metrics_mgr.add_metrics({f"val/{tag}/{k}": v for k, v in response_repetition_metrics(
                    group.batch["input_ids"], group.batch["response_mask"], self.tokenizer.all_special_ids
                ).items()})
        defer.callback(lambda b=batch: DataProto.drop(b))
        scheduler_metrics = reduce_metrics(batch.meta_info.pop("metrics", {}))
        val_metrics_mgr.add_metrics({f"val/{key}": value for key, value in scheduler_metrics.items()})

        metric_keys = [
            key
            for key in (
                "scores",
                "rubric_unweighted_present_rate",
                "rubric_all_criteria_satisfied",
                "rubric_negative_trigger_count",
                "rubric_parse_ok",
                "llmeval_score_sum",
                "llmeval_usable_count",
                "llmeval_question_count",
                "llmeval_judge_success_count",
            )
            if key in batch.batch
        ]
        epoch_batch = batch.pop(batch_keys=metric_keys, non_tensor_batch_keys=["tag"])
        from roll.utils.evaluation_protocol import aggregate_metrics
        rows = [{"tag": str(epoch_batch.non_tensor_batch["tag"][i]),
                 "metrics": {key: float(value[i]) for key, value in epoch_batch.batch.items()}}
                for i in range(len(epoch_batch))]
        evaluation_metrics = aggregate_metrics(rows)
        # LLMEVAL_100_POINT_V1: shared aggregation reports successful turns out of 100.
        llmeval_totals = {"question_count": evaluation_metrics.get("val/llmeval/judge_expected_count", 0)}
        if llmeval_totals["question_count"]:
            if llmeval_totals["question_count"] != 667:
                raise ValueError(f"LLMEval-Med expected 667 questions, found {llmeval_totals['question_count']:g}.")
        for name, value in evaluation_metrics.items():
            val_metrics_mgr.add_metric(name, value)

        defer.close()
        return val_metrics_mgr.get_metrics()
