import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider


_JSON_FENCE_PATTERN = re.compile(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", re.DOTALL | re.IGNORECASE)


class JudgeWhitespaceDegeneration(ValueError):
    """A stream stopped making JSON progress; retry without growing its budget."""


class _JsonWhitespaceGuard:
    def __init__(self, limit):
        self.limit = limit
        self.in_string = False
        self.escaped = False
        self.run = 0

    def feed(self, text):
        if not self.limit:
            return
        for char in text:
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                self.run = 0
            elif char == '"':
                self.in_string = True
                self.run = 0
            elif char in " \t\r\n":
                self.run += 1
                if self.run >= self.limit:
                    raise JudgeWhitespaceDegeneration("Judge emitted excessive consecutive JSON whitespace")
            else:
                self.run = 0


@dataclass
class RubricEvaluation:
    score: float
    unweighted_present_rate: float
    all_criteria_satisfied: float
    negative_trigger_count: float
    attempts: int
    latency: float
    prompt_tokens: int
    completion_tokens: int
    rate_limited_count: int
    server_error_count: int
    success: bool
    format_error_count: int = 0
    truncated_count: int = 0
    judgments: Optional[List[bool]] = None
    early_stop: bool = False
    usage_available: bool = False
    whitespace_abort_count: int = 0


class RubricEvaluatorRewardWorker(Worker):
    """Evaluate generated responses against weighted rubrics using a remote judge."""

    judge_tags = frozenset({"rubric_eval", "healthbench_eval", "rar_science_eval", "researchqa_eval"})
    skip_failed_training_judge = False

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.actor_tokenizer = None
        self._client = None
        self._semaphore = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)
        required_values = {
            "rubric_judge_prompt": self.worker_config.rubric_judge_prompt,
            "rubric_judge_api_base": self.worker_config.rubric_judge_api_base,
            "rubric_judge_model": self.worker_config.rubric_judge_model,
        }
        missing = [name for name, value in required_values.items() if not value]
        if missing:
            raise ValueError(f"Missing rubric evaluator configuration: {missing}")
        if self.worker_config.rubric_judge_concurrency <= 0:
            raise ValueError("rubric_judge_concurrency must be positive")
        if self.worker_config.rubric_judge_timeout <= 0:
            raise ValueError("rubric_judge_timeout must be positive")
        if self.worker_config.rubric_judge_max_retries <= 0:
            raise ValueError("rubric_judge_max_retries must be positive")
        if self.worker_config.rubric_judge_max_tokens <= 0:
            raise ValueError("rubric_judge_max_tokens must be positive")
        if self.worker_config.rubric_judge_retry_max_tokens < self.worker_config.rubric_judge_max_tokens:
            raise ValueError("rubric_judge_retry_max_tokens must be >= rubric_judge_max_tokens")
        if getattr(self.worker_config, "rubric_judge_output_mode", "json_schema") not in ("json_schema", "compact_regex"):
            raise ValueError("rubric_judge_output_mode must be json_schema or compact_regex")
        if getattr(self.worker_config, "rubric_judge_whitespace_limit", 64) < 0:
            raise ValueError("rubric_judge_whitespace_limit must be nonnegative")
        api_key_env = self.worker_config.rubric_judge_api_key_env
        if not os.environ.get(api_key_env):
            raise ValueError(f"Missing rubric judge API key environment variable: {api_key_env}")

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=os.environ[self.worker_config.rubric_judge_api_key_env],
                base_url=self.worker_config.rubric_judge_api_base,
                timeout=self.worker_config.rubric_judge_timeout,
                max_retries=0,
            )
            self._semaphore = asyncio.Semaphore(self.worker_config.rubric_judge_concurrency)

    @staticmethod
    def _parse_judgment(content: str, rubric_count: int) -> List[bool]:
        """Validate the shared complete numbered judgment format."""
        from roll.utils.evaluation_protocol import parse_judgments
        return parse_judgments(content, rubric_count)

    @staticmethod
    def _score_judgments(judgments: List[bool], weights: List[Optional[float]],
                         clip_score: bool = True) -> Dict[str, float]:
        """Compute signed rubric scores using the shared scoring implementation."""
        from roll.utils.evaluation_protocol import score_judgments
        return score_judgments(judgments, weights, clip_score)

    def _format_prompt(self, prompt_str: str, response: str, rubrics_str: str) -> str:
        """Use the shared versioned signed rubric protocol."""
        from roll.utils.evaluation_protocol import format_rubric_prompt
        return format_rubric_prompt(prompt_str, response, rubrics_str)

    @staticmethod
    def _response_format(rubric_count: int):
        if rubric_count <= 0:
            raise ValueError("rubric_count must be positive")
        keys = [str(index) for index in range(1, rubric_count + 1)]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_judgment", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {key: {"type": "string", "enum": ["PRESENT", "NOT_PRESENT"]} for key in keys},
                    "required": keys, "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _compact_json_regex(rubric_count: int) -> str:
        if rubric_count <= 0:
            raise ValueError("rubric_count must be positive")
        # Fixed numeric key order, all keys exactly once, unchanged value choices.
        # No whitespace transitions exist, including before/after the object.
        return r"\{" + ",".join(
            f'"{index}":"(PRESENT|NOT_PRESENT)"' for index in range(1, rubric_count + 1)
        ) + r"\}"

    async def _evaluate(self, prompt_str: str, response: str, rubrics_str: str, weights, rubric_count: int,
                        temperature: float = 0.0):
        self._ensure_client()
        prompt = self._format_prompt(prompt_str, response, rubrics_str)
        output_kwargs = {}
        extra_body = {"chat_template_kwargs": {"enable_thinking": self.worker_config.rubric_judge_enable_thinking}}
        if getattr(self.worker_config, "rubric_judge_structured_output", True):
            mode = getattr(self.worker_config, "rubric_judge_output_mode", "json_schema")
            if mode == "compact_regex":
                extra_body["structured_outputs"] = {"regex": self._compact_json_regex(rubric_count)}
            elif mode == "json_schema":
                output_kwargs["response_format"] = self._response_format(rubric_count)
            else:
                raise ValueError(f"Unknown rubric_judge_output_mode: {mode}")
        prompt += (
            "\n\nReturn only one compact, single-line bare JSON object, with no unnecessary whitespace. "
            "Do not use Markdown fences, explanations or trailing whitespace. "
            "Include every rubric number exactly once. The rubric scoring rules above are unchanged."
        )
        started_at = time.monotonic()
        last_error = None
        rate_limited_count = 0
        server_error_count = 0
        format_error_count = 0
        truncated_count = 0
        whitespace_abort_count = 0
        max_tokens = self.worker_config.rubric_judge_max_tokens
        token_cap = getattr(self.worker_config, "rubric_judge_retry_max_tokens", max(2048, max_tokens))
        for attempt in range(1, self.worker_config.rubric_judge_max_retries + 1):
            content = ""
            finish_reason = None
            received_response = False
            requested_max_tokens = max_tokens
            try:
                usage = None
                judgments = None
                early_stop = False
                whitespace_guard = _JsonWhitespaceGuard(getattr(self.worker_config, "rubric_judge_whitespace_limit", 64))

                async def read_response() -> None:
                    """Close the HTTP stream as soon as a fully validated judgment is available."""
                    nonlocal content, finish_reason, received_response, usage, judgments, early_stop
                    stream = await self._client.chat.completions.create(
                        model=self.worker_config.rubric_judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        stream_options={"include_usage": True},
                        temperature=temperature,
                        seed=getattr(self.worker_config, "rubric_judge_seed", 0),
                        max_tokens=requested_max_tokens,
                        extra_body=extra_body,
                        **output_kwargs,
                    )
                    try:
                        async for chunk in stream:
                            received_response = True
                            if getattr(chunk, "usage", None) is not None:
                                usage = chunk.usage
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            delta = choice.delta.content or ""
                            content += delta
                            finish_reason = choice.finish_reason or finish_reason
                            # Try only at possible object/fence boundaries. The strict parser
                            # validates ALL keys and values, duplicate keys and trailing text.
                            # Incomplete JSON is never repaired or accepted.
                            if content.rstrip().endswith(("}", "```")):
                                try:
                                    judgments = self._parse_judgment(content, rubric_count)
                                except json.JSONDecodeError:
                                    pass
                                else:
                                    early_stop = finish_reason is None
                                    return
                            whitespace_guard.feed(delta)
                    finally:
                        await stream.close()

                async with self._semaphore:
                    # SDK timeouts apply to individual reads. Bound the whole request too,
                    # so a stream of whitespace cannot keep the client alive indefinitely.
                    await asyncio.wait_for(
                        read_response(), timeout=getattr(self.worker_config, "rubric_judge_timeout", 60)
                    )
                if judgments is None:
                    if finish_reason == "length":
                        truncated_count += 1
                        max_tokens = min(token_cap, max_tokens * 2)
                        raise ValueError("Judge output reached token limit before a valid judgment")
                    judgments = self._parse_judgment(content, rubric_count)
                score = self._score_judgments(judgments, list(weights))
                return RubricEvaluation(
                    **score,
                    attempts=attempt,
                    latency=time.monotonic() - started_at,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    rate_limited_count=rate_limited_count,
                    server_error_count=server_error_count,
                    success=True,
                    format_error_count=format_error_count,
                    truncated_count=truncated_count,
                    judgments=judgments,
                    early_stop=early_stop,
                    usage_available=usage is not None,
                    whitespace_abort_count=whitespace_abort_count,
                )
            except Exception as error:
                last_error = error
                from roll.utils.evaluation_protocol import safe_error
                self.logger.warning("Judge request diagnostic: %s", safe_error(error))
                whitespace_abort_count += int(isinstance(error, JudgeWhitespaceDegeneration))
                if received_response:
                    format_error_count += int(finish_reason != "length")
                    # Bounded repr prevents multiline log injection; never log the
                    # prompt, model answer, authorization headers or API error body.
                    self.logger.warning(
                        "Rubric judge invalid output: attempt=%s/%s error=%s detail=%r finish_reason=%s "
                        "rubric_count=%s max_tokens=%s chars=%s head=%r tail=%r",
                        attempt, self.worker_config.rubric_judge_max_retries, type(error).__name__, str(error)[:160],
                        finish_reason, rubric_count, requested_max_tokens, len(content),
                        content[:256], content[-256:] if len(content) > 256 else "",
                    )
                status_code = getattr(error, "status_code", None)
                rate_limited_count += int(status_code == 429)
                server_error_count += int(status_code is not None and status_code >= 500)
                retryable = (
                    isinstance(error, ValueError)
                    or status_code is None
                    or status_code == 429
                    or status_code >= 500
                )
                if not retryable:
                    break
                if attempt < self.worker_config.rubric_judge_max_retries:
                    await asyncio.sleep(2 ** (attempt - 1) + random.random())
        self.logger.warning("Rubric judge request failed after %s attempts: %s", attempt, type(last_error).__name__)
        return RubricEvaluation(
            score=0.0,
            unweighted_present_rate=0.0,
            all_criteria_satisfied=0.0,
            negative_trigger_count=0.0,
            attempts=attempt,
            latency=time.monotonic() - started_at,
            prompt_tokens=0,
            completion_tokens=0,
            rate_limited_count=rate_limited_count,
            server_error_count=server_error_count,
            success=False,
            format_error_count=format_error_count,
            truncated_count=truncated_count,
            whitespace_abort_count=whitespace_abort_count,
        )

    @staticmethod
    def _empty_evaluation() -> RubricEvaluation:
        return RubricEvaluation(
            score=0.0,
            unweighted_present_rate=0.0,
            all_criteria_satisfied=0.0,
            negative_trigger_count=0.0,
            attempts=0,
            latency=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            rate_limited_count=0,
            server_error_count=0,
            success=True,
        )

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    async def compute_rewards(self, data: DataProto):
        tags = data.non_tensor_batch["tag"]
        evaluations = [self._empty_evaluation() for _ in tags]
        pending_indices = [
            index for index, tag in enumerate(tags) if tag in self.judge_tags
        ]
        pending = []
        decoded_responses = {}
        judge_rubrics = {}
        if pending_indices:
            index_tensor = torch.tensor(pending_indices, device=data.batch["responses"].device)
            response_texts = self.actor_tokenizer.batch_decode(
                data.batch["responses"].index_select(0, index_tensor),
                skip_special_tokens=True,
            )
            for index, response in zip(pending_indices, response_texts):
                decoded_responses[index] = response
                rubrics = data.non_tensor_batch["rubrics_str"][index]
                weights = data.non_tensor_batch["rubric_weights"][index]
                rubric_count = int(data.non_tensor_batch["rubric_count"][index])
                judge_kwargs = {}
                config = getattr(self, "pipeline_config", None)
                if tags[index] == "rubric_opsd_train":
                    if getattr(config, "rubric_train_positive_only", False):
                        raise ValueError("RELAY training requires the full signed rubric.")
                    judge_kwargs["temperature"] = getattr(config, "rubric_train_judge_temperature", 0.1)
                # Training and evaluation both keep the original signed rubrics. Do not mutate
                # source columns shared with rollout inputs or later evaluations.
                judge_rubrics[index] = (rubrics, weights)
                pending.append(
                    self._evaluate(
                        prompt_str=data.non_tensor_batch["prompt_str"][index],
                        response=response,
                        rubrics_str=rubrics,
                        weights=weights,
                        rubric_count=rubric_count,
                        **judge_kwargs,
                    )
                )
        if pending:
            results = await asyncio.gather(*pending)
            for index, result in zip(pending_indices, results):
                evaluations[index] = result

        # RuscaRL's training reward retains negative normalized
        # scores. Other training algorithms retain their clipped protocol.
        # HealthBench's separate aggregation protocol is applied below.
        if getattr(getattr(self, "pipeline_config", None), "use_ruscarl", False):
            for index, (tag, evaluation) in enumerate(zip(tags, evaluations)):
                if tag == "rubric_opsd_train" and evaluation.success:
                    evaluation.score = self._score_judgments(
                        evaluation.judgments, list(data.non_tensor_batch["rubric_weights"][index]),
                        clip_score=False,
                    )["score"]

        # HEALTHBENCH_AGGREGATE_CLIP_V1: retain signed per-question scores.
        for index, (tag, evaluation) in enumerate(zip(tags, evaluations)):
            if tag == "healthbench_eval" and evaluation.success:
                weights = data.non_tensor_batch["rubric_weights"][index]
                evaluation.score = sum(
                    weight * present for weight, present in zip(weights, evaluation.judgments)
                    if weight is not None
                ) / sum(weight for weight in weights if weight is not None and weight > 0)

        def tensor(field: str):
            return torch.tensor([getattr(item, field) for item in evaluations], dtype=torch.float32)

        scores = tensor("score")
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.zeros_like(data.batch["responses"], dtype=torch.float16),
                "response_level_rewards": scores,
                "scores": scores,
                "rubric_unweighted_present_rate": tensor("unweighted_present_rate"),
                "rubric_all_criteria_satisfied": tensor("all_criteria_satisfied"),
                "rubric_negative_trigger_count": tensor("negative_trigger_count"),
                "rubric_parse_ok": tensor("success"),
                "llmeval_score_sum": torch.zeros_like(scores),
                "llmeval_usable_count": torch.zeros_like(scores),
                "llmeval_question_count": torch.zeros_like(scores),
                "llmeval_judge_success_count": torch.zeros_like(scores),
            }
        )
        if (getattr(getattr(self, "pipeline_config", None), "opsd_zero_min_unsatisfied_positive", 0)
                and all(tag == "rubric_opsd_train" for tag in tags)):
            from roll.utils.grpo_opsd import unsatisfied_positive_count
            output.batch["rubric_unsatisfied_positive_count"] = torch.tensor([
                unsatisfied_positive_count(data.non_tensor_batch["rubric_weights"][index],
                                           evaluation.judgments, evaluation.success)
                for index, evaluation in enumerate(evaluations)
            ], dtype=torch.long)

        # Keep evaluation tensor schemas compatible with the other reward workers.
        # Only training emits this marker; the rollout loop drops the entire group
        # before any invalid score can enter the replay buffer.
        if self.skip_failed_training_judge and any(tag == "rubric_opsd_train" for tag in tags):
            output.batch["rubric_training_judge_failed"] = torch.tensor(
                [tag == "rubric_opsd_train" and not ev.success for tag, ev in zip(tags, evaluations)],
                dtype=torch.bool,
            )
        if (getattr(getattr(self, "pipeline_config", None), "use_cripo", False)
                and all(tag == "rubric_opsd_train" for tag in tags)):
            import numpy as np
            from roll.utils.relay_only import rubric_metadata

            output.non_tensor_batch["cripo_rubrics"] = np.array([
                rubric_metadata(*judge_rubrics[index], evaluation.judgments)
                if evaluation.success else None
                for index, evaluation in enumerate(evaluations)
            ], dtype=object)
        if isinstance(self, RubricRLSDRewardWorker) and all(tag == "rubric_opsd_train" for tag in tags):
            import numpy as np
            from roll.utils.relay_only import privileged_criteria, privileged_criteria_items

            mode = getattr(getattr(self, "pipeline_config", None), "rubric_rlsd_criteria_mode", "satisfied_positive")
            output.non_tensor_batch["rlsd_criteria"] = np.array([
                privileged_criteria(
                    *judge_rubrics[index],
                    evaluation.judgments,
                    mode=mode,
                ) if evaluation.success else ""
                for index, evaluation in enumerate(evaluations)
            ], dtype=object)
            if mode != "all":
                # Sentence-judge RLSD scores each criterion separately; JSON keeps multi-line bodies intact.
                output.non_tensor_batch["rlsd_criteria_items"] = np.array([
                    json.dumps(privileged_criteria_items(*judge_rubrics[index], evaluation.judgments, mode=mode))
                    if evaluation.success else "[]"
                    for index, evaluation in enumerate(evaluations)
                ], dtype=object)
            if getattr(getattr(self, "pipeline_config", None), "rubric_rlsd_signal", "log_ratio") == "judge_support":
                # Separate request: the reward judgment above is unchanged and identical to GRPO.
                lengths = data.batch["response_mask"].sum(-1).tolist()
                labels = await asyncio.gather(*[
                    self._label_support(
                        data.non_tensor_batch["prompt_str"][index],
                        data.batch["responses"][index, :int(lengths[index])].tolist(),
                        json.loads(output.non_tensor_batch["rlsd_criteria_items"][index]),
                    ) if evaluation.success else asyncio.sleep(0, result=json.dumps({"status": "failed"}))
                    for index, evaluation in enumerate(evaluations)
                ])
                output.non_tensor_batch["rlsd_support"] = np.array(labels, dtype=object)
        if isinstance(self, OPSDFeedbackRewardWorker) and all(tag == "rubric_opsd_train" for tag in tags):
            import numpy as np
            from roll.datasets.chat_template import get_chat_template
            from roll.utils.relay_only import partition_positive_criteria, render_feedback_messages

            config = self.pipeline_config
            template = get_chat_template(config.global_template, self.actor_tokenizer)
            prompt_ids = np.empty(len(tags), dtype=object)
            for index, evaluation in enumerate(evaluations):
                prompt_ids[index] = []
                if not evaluation.success:
                    continue
                met, unmet = partition_positive_criteria(
                    data.non_tensor_batch['rubrics_str'][index],
                    data.non_tensor_batch['rubric_weights'][index], evaluation.judgments,
                    include_met=getattr(config, 'opsd_feedback_include_met', False),
                    include_negative=getattr(config, 'opsd_feedback_include_negative', False),
                )
                messages = render_feedback_messages(
                    data.non_tensor_batch[config.actor_train.data_args.messages][index],
                    decoded_responses[index], met, unmet, config.opsd_feedback_prompt_template,
                )
                ids = self.actor_tokenizer(template(messages, enable_thinking=config.opsd_teacher_thinking))['input_ids']
                # Reserve the entire student continuation; never silently truncate feedback.
                response_length = int(data.batch['response_mask'][index].sum())
                if len(ids) + response_length > config.opsd_feedback_sequence_length:
                    output.batch['rubric_training_judge_failed'][index] = True
                    continue
                prompt_ids[index] = ids
            output.non_tensor_batch['opsd_feedback_teacher_prompt_ids'] = prompt_ids
        request_count = len(pending)
        successful_requests = sum(evaluations[index].success for index in pending_indices)
        total_latency = sum(evaluations[index].latency for index in pending_indices)
        output.meta_info = {
            "metrics": {
                "rubric_judge/request_count@sum": request_count,
                "rubric_judge/early_stop_count@sum": sum(evaluations[i].early_stop for i in pending_indices),
                "rubric_judge/whitespace_abort_count@sum": sum(
                    evaluations[i].whitespace_abort_count for i in pending_indices
                ),
                "rubric_judge/usage_available_count@sum": sum(
                    evaluations[i].usage_available for i in pending_indices
                ),
                "rubric_judge/success_count@sum": successful_requests,
                "rubric_judge/failure_count@sum": request_count - successful_requests,
                "rubric_judge/retry_count@sum": sum(
                    max(evaluations[index].attempts - 1, 0) for index in pending_indices
                ),
                "rubric_judge/latency_seconds@mean": total_latency / request_count if request_count else 0.0,
                "rubric_judge/prompt_tokens@sum": sum(
                    evaluations[index].prompt_tokens for index in pending_indices
                ),
                "rubric_judge/completion_tokens@sum": sum(
                    evaluations[index].completion_tokens for index in pending_indices
                ),
                "rubric_judge/http_429_count@sum": sum(
                    evaluations[index].rate_limited_count for index in pending_indices
                ),
                "rubric_judge/http_5xx_count@sum": sum(
                    evaluations[index].server_error_count for index in pending_indices
                ),
                "rubric_judge/format_error_count@sum": sum(
                    evaluations[index].format_error_count for index in pending_indices
                ),
                "rubric_judge/truncated_count@sum": sum(
                    evaluations[index].truncated_count for index in pending_indices
                ),
            }
        }
        return output


class RubricRLRewardWorker(RubricEvaluatorRewardWorker):
    """Score training rollouts as well as evaluation; OPSD's worker stays eval-only."""

    judge_tags = RubricEvaluatorRewardWorker.judge_tags | {"rubric_opsd_train"}
    skip_failed_training_judge = True


class RubricRLSDRewardWorker(RubricRLRewardWorker):
    """Build privileged criteria without changing the judgment used for GRPO rewards."""

    def _support_client(self):
        """Support labelling may use its own endpoint so it does not share the reward judge's load."""
        self._ensure_client()
        base = getattr(self.pipeline_config, "rubric_rlsd_support_api_base", None)
        if not base or base == self.worker_config.rubric_judge_api_base:
            return self._client
        if getattr(self, "_support_client_instance", None) is None:
            from openai import AsyncOpenAI

            self._support_client_instance = AsyncOpenAI(
                api_key=os.environ[self.worker_config.rubric_judge_api_key_env], base_url=base,
                timeout=self.worker_config.rubric_judge_timeout, max_retries=0,
            )
        return self._support_client_instance

    async def _label_support(self, question: str, token_ids: List[int], criteria: List[str]) -> str:
        """Separate judge call citing the sentences that satisfy each met criterion (JSON record)."""
        from roll.utils.relay_only import parse_support, response_sentences, support_request

        if not criteria:
            return json.dumps({"status": "no_criteria"})
        segments, texts = response_sentences(
            token_ids, self.actor_tokenizer, getattr(self.pipeline_config, "rubric_rlsd_segment_min_tokens", 4))
        if not texts:
            return json.dumps({"status": "no_criteria"})
        client = self._support_client()
        prompt, response_format = support_request(question, texts, criteria)
        for attempt in range(1, self.worker_config.rubric_judge_max_retries + 1):
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(client.chat.completions.create(
                        model=self.worker_config.rubric_judge_model,
                        messages=[{"role": "user", "content": prompt}], temperature=0.0,
                        seed=getattr(self.worker_config, "rubric_judge_seed", 0),
                        max_tokens=getattr(self.pipeline_config, "rubric_rlsd_support_max_tokens", 4096),
                        response_format=response_format,
                        extra_body={"chat_template_kwargs": {
                            "enable_thinking": self.worker_config.rubric_judge_enable_thinking}},
                    ), timeout=getattr(self.worker_config, "rubric_judge_timeout", 60))
                support = parse_support(response.choices[0].message.content, len(criteria), len(texts))
                return json.dumps({"status": "ok", "segments": segments, "support": support})
            except Exception as error:
                from roll.utils.evaluation_protocol import safe_error
                self.logger.warning("RLSD support labelling attempt %s failed: %s", attempt, safe_error(error))
                if attempt < self.worker_config.rubric_judge_max_retries:
                    await asyncio.sleep(2 ** (attempt - 1) + random.random())
        # A failed labelling falls back to GRPO for this response; it is never read as "no support".
        return json.dumps({"status": "failed"})


class OPSDFeedbackRewardWorker(RubricRLRewardWorker):
    """Judge each student rollout and construct its positive-only teacher feedback."""
