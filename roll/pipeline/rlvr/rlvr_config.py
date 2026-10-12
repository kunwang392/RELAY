import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from roll.configs.base_config import PPOConfig, RouterArguments
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger


logger = get_logger()


@dataclass
class DatasetFilterConfig:
    source: Optional[str] = None
    min_difficulty: Optional[float] = None
    max_difficulty: Optional[float] = None
    num_samples: int = 0

@dataclass
class RewardFilterConfig:
    type: Literal["no_filter", "mean_filter", "std_filter"] = field(
        default="no_filter",
        metadata={"help": "Type of filter to apply to rewards."},
    )
    filter_args: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Arguments used in `filter_fn`"},
    )


@dataclass
class RewardConfig(WorkerConfig):
    code_url: str = field(
        default=None,
        metadata={"help": "The url of the code."}
    )
    use_local: bool = field(
        default=True,
        metadata={"help": "Whether to use local code instead of downloading from URL."}
    )
    judge_prompt: str = field(
        default=None,
        metadata={"help": "The prompt for judge."}
    )
    judge_model_type: str = field(
        default=None,
        metadata={"help": "api or inference"}
    )
    judge_model_name: str = field(
        default=None,
        metadata={"help": "judge_model_name."}
    )
    judge_api_url: str = field(
        default=None,
        metadata={"help": "judge_api_url."}
    )
    judge_api_key: str = field(
        default=None,
        metadata={"help": "judge_api_key."}
    )
    rubric_judge_prompt: str = field(
        default=None,
        metadata={"help": "Prompt template for rubric evaluation."},
    )
    rubric_judge_api_base: str = field(
        default=None,
        metadata={"help": "OpenAI-compatible API base URL for rubric evaluation."},
    )
    rubric_judge_model: str = field(
        default=None,
        metadata={"help": "Remote rubric judge model name."},
    )
    rubric_judge_api_key_env: str = field(
        default="RUBRIC_JUDGE_API_KEY",
        metadata={"help": "Environment variable containing the rubric judge API key."},
    )
    rubric_judge_concurrency: int = field(
        default=32,
        metadata={"help": "Maximum concurrent rubric judge requests per worker."},
    )
    rubric_judge_timeout: float = field(
        default=60.0,
        metadata={"help": "Rubric judge request timeout in seconds."},
    )
    rubric_judge_max_retries: int = field(
        default=3,
        metadata={"help": "Maximum rubric judge request attempts."},
    )
    rubric_judge_retry_max_tokens: int = field(
        default=2048, metadata={"help": "Maximum rubric judge output budget after truncation retries."}
    )
    rubric_judge_structured_output: bool = field(
        default=True, metadata={"help": "Constrain rubric judge output to the complete numbered JSON schema."}
    )
    rubric_judge_output_mode: str = field(
        default="json_schema",
        metadata={"help": "json_schema or compact_regex (vLLM structured_outputs; forbids JSON whitespace)."},
    )
    rubric_judge_whitespace_limit: int = field(
        default=64,
        metadata={"help": "Abort after this many consecutive JSON whitespace characters outside strings; 0 disables."},
    )
    rubric_judge_max_tokens: int = field(
        default=512,
        metadata={"help": "Maximum completion tokens for a rubric judge response."},
    )
    rubric_judge_enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for the rubric judge."},
    )
    rubric_judge_seed: int = field(
        default=0, metadata={"help": "Fixed request seed; does not guarantee batch-invariant server inference."},
    )
    format_pattern: str = field(
        default=None,
        metadata={"help": "The pattern of the answer format."}
    )
    reward_type: str = field(default=None, metadata={"help": "The type of the reward."})
    response_length_penalty_coef: float = field(default=0.0, metadata={"help": "The coefficient of the response length penalty."})

    tag_included: List[str] = field(default_factory=list, metadata={"help": "The tags of the domain."})
    query_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward query filtering"},)
    response_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward response filtering"},
    )
    
    reward_manager_config: List[Dict[str, Any]] = field(default_factory=list, metadata={"help": "The reward system plugin config."})


@dataclass
class RLVRConfig(PPOConfig):
    use_ruscarl: bool = False
    ruscarl_clip_ratio_c: float = 3.0
    ruscarl_sigmoid_midpoint: float = 0.20
    ruscarl_sigmoid_steepness: float = 125.0
    ruscarl_max_prompt_length: int = 12288  # Rollout-only; original training prompt budget is unchanged.
    # Shared rubric GRPO/RLSD controls; enabled by the rubric experiment defaults.
    rubric_train_positive_only: bool = False
    rubric_train_judge_temperature: float = 0.0
    grpo_ppo_mini_batch_size: Optional[int] = None  # Complete prompt groups per update.

    use_cripo: bool = False
    cripo_behavior_injection: bool = True
    cripo_advantage_correction: bool = True
    cripo_beta: float = 0.05
    cripo_gamma: float = 0.95
    cripo_alpha: float = 0.1
    cripo_tau_flip: float = 0.1
    cripo_temperature: float = 1.0
    cripo_decay_fraction: float = 0.25
    cripo_boundary_fraction: float = 0.01
    cripo_kl_cap: float = 10.0
    cripo_kappa: int = 2048
    cripo_zeta: int = 10
    cripo_max_unexplored: int = 3
    cripo_teacher_max_length: int = 12288
    cripo_kl_chunk_size: int = 128
    cripo_distillation_topk: int = 64
    cripo_distillation_add_tail: bool = True  # SDPO default; CriPO does not specify tail handling.
    cripo_ppo_mini_batch_size: int = 32  # Prompt groups per optimizer update.
    cripo_judge_temperature: float = 0.1  # Applied only to CriPO training judgments.
    use_rubric_rlsd: bool = False
    rubric_rlsd_clip: Optional[float] = 0.2  # None disables evidence-ratio clipping; 0 disables reweighting.
    rubric_rlsd_strength: float = 1.0
    rubric_rlsd_decay_steps: int = 0
    rubric_rlsd_plain_criteria: bool = False
    rubric_rlsd_criteria_mode: Literal["satisfied_positive", "unsatisfied_positive", "all"] = "satisfied_positive"
    # Sentence granularity shares one summed/mean evidence ratio across each sentence (step-level SSPO).
    rubric_rlsd_granularity: Literal["token", "sentence"] = "token"
    rubric_rlsd_segment_reduce: Literal["sum", "mean"] = "sum"
    rubric_rlsd_segment_min_tokens: int = 4
    rubric_rlsd_upper_clip_only: bool = False  # Bound only amplification, i.e. min(ratio, 1 + clip).
    rubric_rlsd_same_teacher_baseline: bool = False  # Score the original context with the teacher too.
    # "sentence_judge": the teacher judges labeled sentences per met criterion (roll/utils/rubric_rlsd_judge.py).
    # "judge_support": the remote judge cites supporting sentences in a separate call (upper-bound control).
    rubric_rlsd_signal: Literal["log_ratio", "sentence_judge", "judge_support"] = "log_ratio"
    rubric_rlsd_support_max_tokens: int = 4096
    # Optional separate endpoint (same judge model) for support labelling; None reuses rubric_judge_api_base.
    rubric_rlsd_support_api_base: Optional[str] = None
    # "redistribute": mean-one up/down weighting; "additive": only supporting sentences move, favourably.
    rubric_rlsd_judge_weighting: Literal["redistribute", "additive"] = "redistribute"
    rubric_rlsd_judge_epsilon: float = 0.3  # w = 1 + sign(A) * eps * tanh(z), mean-one per response.
    rubric_rlsd_judge_fills: List[str] = field(default_factory=lambda: ["Yes", "No"])
    rubric_rlsd_judge_buckets: int = 4  # Length buckets for judge forwards.
    rubric_rlsd_judge_prompt: Optional[str] = None  # None uses rubric_rlsd_judge.JUDGE_PROMPT.
    rubric_rlsd_reweight_negative: bool = False
    rubric_rlsd_negative_only: bool = False
    rubric_rlsd_invert_negative_ratio: bool = True  # Preserve legacy signed-mode Ps/Pt on negatives.
    rubric_rlsd_normalize_weights: bool = False
    rubric_rlsd_amplify_only: bool = False
    rubric_rlsd_token_fraction: float = 1.0  # Per-trajectory top absolute log-ratio fraction; 1 keeps dense RLSD.
    rubric_rlsd_negative_protection: bool = False
    rubric_rlsd_signal_coverage: Optional[float] = None
    rubric_rlsd_invert_positive_ratio: bool = False  # Use Ps/Pt for positive advantages before clipping.
    rubric_rlsd_teacher_mode: Literal["current", "student_sync", "frozen"] = "current"
    rubric_rlsd_teacher_update_frequency: int = 10
    rubric_rlsd_teacher: Optional[WorkerConfig] = None
    rubric_rlsd_prompt_template: str = (
        "When answering the question, satisfy the following requirements:\n{criteria}"
    )
    # global
    train_data_shuffle_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Training dataset shuffle seed; None uses the global seed. Validation is unaffected."},
    )
    global_template: str = field(
        default=None,
        metadata={"help": "The template of the global."})
    tag_to_template: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Mapping from tag to template name. Tags not in this dict fall back to global_template."}
    )
    dataset_filter: DatasetFilterConfig = field(
        default_factory=DatasetFilterConfig,
        metadata={"help": "Configuration for filtering dataset by source and difficulty"},
    )
    num_return_sequences_in_group: int = field(
        default=1,
        metadata={"help": "The number of return sequences in one group, used in generation_args."}
    )

    save_logging_board_dir: str = field(
        default=None, metadata={"help": "saving directory of logging board_metrics"}
    )

    # role related
    validation: WorkerConfig = field(
        default=None,
        metadata={"help": "Configuration for the validation."}
    )
    rewards: Optional[Dict[str, RewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for the multi domain rewards."}
    )
    reward_model: Optional[WorkerConfig] = field(
        default=None,
        metadata={"help": "Configuration for the shared reward model cluster (InferWorker + vLLM)."}
    )

    # PPO related
    difficulty_loss_weight: bool = field(default=False, metadata={"help": "Use difficulty_loss_weight"})
    length_loss_weight: bool = field(default=False, metadata={"help": "Use length_loss_weight"})
    postive_loss_coef: float = field(
        default=0,
        metadata={"help": "Loss coefficient for SFT loss, used for positive samples"}
    )
    use_topr_neg_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Loss coefficient for TOPR Neg loss"}
    )
    use_policy_loss_type: Literal["PPO", "PG"] = field(
        default="PPO",
        metadata={"help": "whether to use PPO/PG loss"}
    )
    use_topr_loss: bool = field(
        default=False,
        metadata={"help": "whether to use TPRO loss, http://arxiv.org/abs/2503.14286"}
    )
    rl_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Loss coefficient for RL loss"}
    )
    importance_sampling: Literal["token", "seq"] = (
        field(default="token", metadata={"help": "policy importance sampling"})
    )

    val_greedy: bool = field(default=False, metadata={"help": "Use greedy for validation"})
    val_n_sample: int = field(default=1, metadata={"help": "Number of samples for validation"})
    max_len_mask: bool = field(default=False)
    mask_type: Literal["all", "loss"] = field(default="loss", metadata={"help": "Mask type: 'all' or 'loss'"})
    difficulty_mask: bool = field(default=False)
    balance_length: bool = field(default=False)
    minibatch_data_iter_num: int = field(default=1)
    difficulty_low_threshold: float = field(default=0.0)
    difficulty_high_threshold: float = field(default=1.0)
    error_max_len_clip: bool = field(default=False)
    error_max_len_threshold: int = field(default=9999999999)

    def _validate_rubric_rlsd(self) -> None:
        """Restrict RLSD to synchronous, group-centered GRPO with same-policy log probabilities."""
        if not self.use_rubric_rlsd:
            return
        if self.is_pure_opd or self.use_opd or self.use_opsd or self.adv_estimator != "grpo":
            raise ValueError("Rubric RLSD requires GRPO with OPD/OPSD disabled.")
        if not self.enable_old_logprobs_recompute or self.async_pipeline or self.async_generation_ratio != 0:
            raise ValueError("Rubric RLSD requires synchronous training and recomputed old log probabilities.")
        if self.actor_train.use_dynamic_batching_in_infer:
            raise ValueError("Rubric RLSD currently requires static inference microbatches.")
        if (self.norm_mean_type != "group" or self.gamma != 1 or self.whiten_advantages
                or self.whiten_rewards or self.add_token_level_kl):
            raise ValueError("Rubric RLSD requires group-centered GRPO without whitening or token KL rewards.")
        if self.num_return_sequences_in_group < 2 or self.tag_to_template:
            raise ValueError("Rubric RLSD requires groups of at least two and a single global chat template.")
        if not self.global_template:
            raise ValueError("Rubric RLSD requires an explicit global_template.")
        if ((self.rubric_rlsd_clip is not None and not 0 <= self.rubric_rlsd_clip < 1)
                or not 0 <= self.rubric_rlsd_strength <= 1):
            raise ValueError("Rubric RLSD requires clip=None or clip in [0, 1), and strength in [0, 1].")
        if self.rubric_rlsd_decay_steps < 0:
            raise ValueError("rubric_rlsd_decay_steps cannot be negative.")
        if getattr(self, "rubric_rlsd_negative_only", False) and self.rubric_rlsd_reweight_negative:
            raise ValueError("RLSD negative_only and reweight_negative are mutually exclusive.")
        fraction = getattr(self, "rubric_rlsd_token_fraction", 1.0)
        if not 0 < fraction <= 1:
            raise ValueError("rubric_rlsd_token_fraction must be in (0, 1].")
        if fraction < 1 and self.rubric_rlsd_normalize_weights:
            raise ValueError("Sparse RLSD must not renormalize selected token weights.")
        protection = getattr(self, "rubric_rlsd_negative_protection", False)
        coverage = getattr(self, "rubric_rlsd_signal_coverage", None)
        if protection and (not (self.rubric_rlsd_negative_only or self.rubric_rlsd_reweight_negative)
                           or not self.rubric_rlsd_invert_negative_ratio or self.rubric_rlsd_invert_positive_ratio
                           or self.rubric_rlsd_normalize_weights or self.rubric_rlsd_amplify_only):
            raise ValueError(
                "Negative protection requires negative trajectories, standard ratios, "
                "and no normalization/amplification."
            )
        if coverage is not None and (not protection or not 0 < coverage <= 1 or fraction != 1):
            raise ValueError("Signal coverage requires negative protection, coverage in (0, 1], and token_fraction=1.")
        if getattr(self, "rubric_rlsd_criteria_mode", "satisfied_positive") not in {
            "satisfied_positive", "unsatisfied_positive", "all",
        }:
            raise ValueError("Unknown rubric_rlsd_criteria_mode.")
        granularity = getattr(self, "rubric_rlsd_granularity", "token")
        if granularity not in {"token", "sentence"}:
            raise ValueError("Unknown rubric_rlsd_granularity.")
        if granularity == "sentence" and (
            getattr(self, "rubric_rlsd_segment_reduce", "sum") not in {"sum", "mean"}
            or getattr(self, "rubric_rlsd_segment_min_tokens", 4) < 1 or fraction != 1 or protection
            or self.rubric_rlsd_normalize_weights or self.rubric_rlsd_amplify_only
        ):
            raise ValueError("Sentence RLSD needs sum/mean reduction, min_tokens >= 1, dense tokens, "
                             "and no protection/normalization/amplification.")
        if getattr(self, "rubric_rlsd_upper_clip_only", False) and self.rubric_rlsd_clip is None:
            raise ValueError("rubric_rlsd_upper_clip_only requires rubric_rlsd_clip.")
        signal = getattr(self, "rubric_rlsd_signal", "log_ratio")
        if signal not in {"log_ratio", "sentence_judge", "judge_support"}:
            raise ValueError("Unknown rubric_rlsd_signal.")
        if signal == "judge_support" and getattr(self, "rubric_rlsd_support_max_tokens", 4096) < 1:
            raise ValueError("rubric_rlsd_support_max_tokens must be positive.")
        if getattr(self, "rubric_rlsd_judge_weighting", "redistribute") not in {"redistribute", "additive"}:
            raise ValueError("Unknown rubric_rlsd_judge_weighting.")
        if getattr(self, "rubric_rlsd_support_api_base", None) and signal != "judge_support":
            raise ValueError("rubric_rlsd_support_api_base only applies to judge_support RLSD.")
        if signal in {"sentence_judge", "judge_support"}:
            fills = list(getattr(self, "rubric_rlsd_judge_fills", ["Yes", "No"]))
            prompt = getattr(self, "rubric_rlsd_judge_prompt", None)
            if (granularity != "sentence" or not 0 < self.rubric_rlsd_judge_epsilon < 1
                    or not fills or not set(fills) <= {"Yes", "No"} or len(set(fills)) != len(fills)
                    or getattr(self, "rubric_rlsd_judge_buckets", 4) < 1
                    or self.rubric_rlsd_criteria_mode == "all" or fraction != 1 or protection
                    or self.rubric_rlsd_normalize_weights or self.rubric_rlsd_amplify_only
                    or self.rubric_rlsd_invert_positive_ratio):
                raise ValueError("Sentence-judge RLSD needs sentence granularity, 0 < epsilon < 1, Yes/No fills, "
                                 "positive criteria items, dense tokens and no log-ratio-only options.")
            if prompt is not None:
                for key in ("{question}", "{sentences}", "{criterion}"):
                    if key not in prompt:
                        raise ValueError(f"rubric_rlsd_judge_prompt must contain {key}.")
                prompt.format(question="q", sentences="s", criterion="c")
            teacher = self.rubric_rlsd_teacher if self.rubric_rlsd_teacher is not None else self.actor_train
            if signal == "sentence_judge" and teacher.strategy_args.strategy_config.get("context_parallel_size", 1) != 1:
                raise ValueError("Sentence-judge RLSD reads per-position logits and requires CP=1.")
        if (getattr(self, "rubric_rlsd_amplify_only", False)
                and self.rubric_rlsd_normalize_weights):
            raise ValueError("RLSD amplify-only weights cannot use trajectory mean-one normalization.")
        if "{criteria}" not in self.rubric_rlsd_prompt_template:
            raise ValueError("rubric_rlsd_prompt_template must contain {criteria}.")
        self.rubric_rlsd_prompt_template.format(criteria="test")

    def _configure_rubric_rlsd_teacher(self) -> None:
        """Configure a privileged scorer independently of the frozen KL reference."""
        if not self.use_rubric_rlsd:
            return
        if self.rubric_rlsd_teacher_mode not in {"current", "student_sync", "frozen"}:
            raise ValueError("Unknown rubric_rlsd_teacher_mode.")
        if self.rubric_rlsd_teacher_update_frequency < 1:
            raise ValueError("rubric_rlsd_teacher_update_frequency must be positive.")
        if self.rubric_rlsd_teacher_mode == "current":
            if self.rubric_rlsd_teacher is not None:
                raise ValueError("Current-policy RLSD must not configure a separate teacher.")
            return
        teacher = self.rubric_rlsd_teacher
        if teacher is None or not teacher.is_configured:
            raise ValueError("Snapshot/frozen RLSD requires rubric_rlsd_teacher configuration.")
        if self.resume_from_checkpoint and self.rubric_rlsd_teacher_mode == "student_sync":
            # A frozen teacher always reloads the initial model, so resuming loses no teacher history.
            raise ValueError("Snapshot RLSD currently requires a fresh run to preserve teacher history.")
        teacher.name = "rubric_rlsd_teacher"
        teacher.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if teacher.model_args.model_name_or_path is None:
            teacher.model_args.model_name_or_path = self.pretrain
        if teacher.model_args.model_name_or_path != self.actor_train.model_args.model_name_or_path:
            raise ValueError("RLSD teacher must start from the same model as the student.")
        if teacher.use_dynamic_batching_in_infer or teacher.use_sequence_packing:
            raise ValueError("RLSD teacher requires static inference without sequence packing.")
        if (self.actor_train.strategy_args.strategy_name != "megatron_train"
                or teacher.strategy_args.strategy_name != "megatron_infer"):
            raise ValueError("Snapshot/frozen RLSD currently requires Megatron train/infer strategies.")
        if self.rubric_rlsd_teacher_mode == "student_sync":
            if teacher.device_mapping != self.actor_train.device_mapping:
                raise ValueError("RLSD snapshots require matching teacher/student device mappings.")
            for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                        "context_parallel_size", "expert_model_parallel_size",
                        "virtual_pipeline_model_parallel_size"):
                if (teacher.strategy_args.strategy_config.get(key, 1)
                        != self.actor_train.strategy_args.strategy_config.get(key, 1)):
                    raise ValueError(f"RLSD snapshots require matching {key}.")

    def _apply_opsd_validation_sampling_profile(self):
        profiles = self.opsd_validation_sampling_profiles
        if not self.use_opsd or not profiles or self.validation is None:
            return

        profile_name = "thinking" if self.opsd_student_thinking else "non_thinking"
        profile = profiles.get(profile_name)
        required_keys = {"temperature", "top_p", "top_k"}
        if profile is None or set(profile.keys()) != required_keys:
            raise ValueError(
                f"opsd_validation_sampling_profiles.{profile_name} must contain exactly "
                f"{sorted(required_keys)}"
            )
        for key, value in profile.items():
            setattr(self.validation.generating_args, key, value)

    def __post_init__(self):
        self._validate_rubric_rlsd()
        # Handle OPD mapping FIRST before any access to actor_train/actor_infer/reference
        # This ensures student_train/student_infer/teacher are mapped correctly
        self._handle_opd_mapping()

        # Now safe to access actor_infer (may have been mapped from student_infer)
        self.actor_infer.generating_args.num_return_sequences = self.num_return_sequences_in_group
        super().__post_init__()
        from roll.utils.relay_only import validate_cripo
        validate_cripo(self)
        from roll.utils.rubric_training import validate_rubric_training
        validate_rubric_training(self)
        from roll.utils.relay_only import validate_ruscarl
        validate_ruscarl(self)
        self._configure_rubric_rlsd_teacher()
        self._apply_opsd_validation_sampling_profile()

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.base_worker.InferWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"
        if self.reward_model is not None and self.reward_model.worker_cls is None:
            self.reward_model.worker_cls = "roll.pipeline.base_worker.InferWorker"
        # Multi-teacher: set worker_cls for each _reference_configs entry
        if hasattr(self, '_reference_configs'):
            for ref_cfg in self._reference_configs.values():
                if ref_cfg.worker_cls is None:
                    ref_cfg.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"

        if self.router_args is None:
            self.router_args = RouterArguments(router_name="PromptAffinityRouter", router_config=dict())
            self.router_args.max_running_requests = self.max_running_requests

        logger.info(f"actor_train.worker_cls: {self.actor_train.worker_cls}")

        self.domain_2_tag = None
        self.tag_2_domain = None
        if self.rewards is not None:
            self.domain_2_tag = {key: set(worker_config.tag_included) for key, worker_config in self.rewards.items()}
            self.tag_2_domain = {
                tag: key for key, worker_config in self.rewards.items() for tag in worker_config.tag_included
            }

        # OPD mode: include teacher tag_included in tag_2_domain mapping
        # so that teacher tags are also mapped to domains for routing
        if self.is_pure_opd or self.use_opd:
            if self.tag_2_domain is None:
                self.tag_2_domain = {}
                self.domain_2_tag = {}
            for name, ref_cfg in self._reference_configs.items():
                for tag in ref_cfg.tag_included:
                    if tag not in self.tag_2_domain:
                        # If no reward covers this tag, map tag to itself as domain
                        self.tag_2_domain[tag] = tag
                        if tag not in self.domain_2_tag:
                            self.domain_2_tag[tag] = {tag}

        # Validate tag_to_template entries
        if self.tag_to_template:
            from roll.datasets.chat_template import chat_templates
            for tag, tmpl_name in self.tag_to_template.items():
                if tmpl_name not in chat_templates:
                    raise ValueError(
                        f"Template '{tmpl_name}' for tag '{tag}' in tag_to_template not found. "
                        f"Available templates: {list(chat_templates.keys())}"
                    )

        if self.async_pipeline:
            assert self.async_generation_ratio >= 1.0, "async_generation_ratio must be >= 1.0"
            infer_devices = self.actor_infer.device_mapping
            other_worker_devices = set()
            for worker_config in [
                self.actor_train,
                self.critic,
                self.rewards,
                self.reference,
            ]:
                if worker_config is None:
                    continue
                if isinstance(worker_config, dict):
                    for config in worker_config.values():
                        other_worker_devices.update(config.device_mapping or set())
                else:
                    other_worker_devices.update(worker_config.device_mapping or set())

            if infer_devices is not None and len(set(infer_devices).intersection(other_worker_devices)) != 0:
                logger.warning("infer worker are sharing devices with other workers, which may cause performance issue")

            assert self.generate_opt_level == 1, "AsyncRLVRPipeline only support generate_opt_level 1"
            if self.num_return_sequences_in_group > 1 and not self.is_num_return_sequences_expand:
                self.is_num_return_sequences_expand = True
                logger.warning("Async Pipeline must is_num_return_sequences_expand is True when num_return_sequences_in_group > 1")

        if self.actor_infer:
            self.actor_infer.generating_args.max_new_tokens = self.sequence_length - self.prompt_length
            logger.warning(f"rewrite actor_infer max_new_tokens: {self.actor_infer.generating_args.max_new_tokens}")
        if self.validation:
            self.validation.generating_args.max_new_tokens = self.val_sequence_length - self.val_prompt_length
            logger.warning(f"rewrite validation max_new_tokens: {self.validation.generating_args.max_new_tokens}")

        # infer the required num nodes
        total_devices = []
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, WorkerConfig):
                if attribute.device_mapping is not None:
                    total_devices.extend(attribute.device_mapping)
        for worker_config in self.rewards.values():
            if worker_config.device_mapping is not None:
                total_devices.extend(worker_config.device_mapping)
        # Also include _reference_configs device_mappings for multi-teacher
        if hasattr(self, '_reference_configs'):
            for ref_cfg in self._reference_configs.values():
                if isinstance(ref_cfg, WorkerConfig) and ref_cfg.device_mapping is not None:
                    total_devices.extend(ref_cfg.device_mapping)
        if len(total_devices) > 0:
            max_gpu_num = max(total_devices) + 1
            if max_gpu_num <= self.num_gpus_per_node:
                self.num_nodes = 1
            else:
                self.num_nodes = (max_gpu_num + self.num_gpus_per_node - 1) // self.num_gpus_per_node

        # Apply OPD configuration at the end (handles student_train/student_infer/teacher mapping)
        self._apply_opd_config()

    def to_dict(self):
        return dataclasses.asdict(self)
