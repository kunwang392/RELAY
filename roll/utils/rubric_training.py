"""Complete-group GRPO minibatches used by RELAY."""

import math
from typing import Any

import torch


def group_update_size(config: Any, dp_size: int) -> int:
    """Validate the physical optimizer batch against its configured prompt count."""
    args = config.actor_train.training_args
    actual = args.per_device_train_batch_size * args.gradient_accumulation_steps * dp_size
    expected = config.grpo_ppo_mini_batch_size * config.num_return_sequences_in_group
    if actual != expected or config.rollout_batch_size % config.grpo_ppo_mini_batch_size:
        raise ValueError(f"Group-preserving GRPO optimizer batch must contain {expected} responses; got {actual}. "
                         "Rollout query count must be divisible by grpo_ppo_mini_batch_size.")
    return expected


def group_minibatch_permutation(lengths: torch.Tensor, dp_size: int, minibatch_size: int) -> torch.Tensor:
    """Balance each complete prompt minibatch separately, then lay out contiguous DP shards.

    Worker shuffling must be disabled so DP-local shuffling cannot mix the
    original prompt minibatches.
    """
    if dp_size < 1 or minibatch_size < 1 or minibatch_size % dp_size or len(lengths) % minibatch_size:
        raise ValueError("Group-preserving GRPO needs complete, equally divisible optimizer minibatches.")
    partitions = [[] for _ in range(dp_size)]
    for start in range(0, len(lengths), minibatch_size):
        order = lengths[start:start + minibatch_size].argsort(descending=True, stable=True) + start
        order = order.reshape(-1, dp_size)
        order[1::2] = order[1::2].flip(-1)
        for rank in range(dp_size):
            partitions[rank].append(order[:, rank])
    return torch.cat([torch.cat(parts) for parts in partitions])


def balance_group_batch(batch: Any, config: Any, dp_size: int) -> dict:
    size = group_update_size(config, dp_size)
    # The pipeline's prompt_id is an original *response row* ordinal, not a query ID.
    ids = (batch.batch["prompt_id"] // config.num_return_sequences_in_group).reshape(
        -1, config.num_return_sequences_in_group)
    if not (ids == ids[:, :1]).all():
        raise ValueError("Group-preserving GRPO requires contiguous complete prompt groups before minibatch balancing.")
    lengths = batch.batch["attention_mask"].sum(-1)
    batch.reorder(group_minibatch_permutation(lengths, dp_size, size))
    batch.pop("prompt_id")
    return {"grpo/optimizer_steps_per_rollout": len(lengths) // size}


def validate_rubric_training(config: Any) -> None:
    temperature = config.rubric_train_judge_temperature
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("rubric_train_judge_temperature must be finite and non-negative.")
    size = config.grpo_ppo_mini_batch_size
    if size is None:
        return
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("grpo_ppo_mini_batch_size must be a positive integer.")
    worker = config.actor_train
    strategy = worker.strategy_args.strategy_config
    if (config.adv_estimator != "grpo" or config.ppo_epochs != 1 or config.async_pipeline
            or worker.use_sequence_packing or worker.use_dynamic_batching_in_train
            or worker.strategy_args.strategy_name != "megatron_train"
            or any(strategy.get(key, 1) not in (None, 1) for key in (
                "pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size",
                "virtual_pipeline_model_parallel_size"))):
        raise ValueError("Group minibatches require synchronous GRPO, dense Megatron TP/DP, no packing, one PPO epoch.")
    tp = strategy.get("tensor_model_parallel_size", 1)
    if len(worker.device_mapping) % tp:
        raise ValueError("Actor device count must be divisible by tensor parallel size.")
    group_update_size(config, len(worker.device_mapping) // tp)
    if config.force_disable_old_logprobs_recompute:
        raise ValueError("Group minibatches require old policy log probabilities.")
    config.enable_old_logprobs_recompute = True
    if config.use_cripo and size != config.cripo_ppo_mini_batch_size:
        raise ValueError("Shared and CriPO prompt minibatch sizes must match.")
