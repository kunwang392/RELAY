"""Configuration and trajectory selection for additive GRPO + forward-KL OPSD."""

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch


def validate_grpo_opsd(config: Any) -> None:
    """Reject combinations that change the meaning of the hybrid objective."""
    if not config.use_opsd or not config.use_opd or config.is_pure_opd:
        raise ValueError("GRPO+OPSD requires use_opsd=True, use_opd=True, is_pure_opd=False.")
    if config.adv_estimator != "grpo" or config.num_return_sequences_in_group < 2:
        raise ValueError("GRPO+OPSD requires grpo and at least two responses per prompt.")
    if config.opsd_loss_type != "forward_kl" or config.opsd_teacher_update_mode != "fixed":
        raise ValueError("GRPO+OPSD requires forward_kl and a fixed teacher.")
    if config.opsd_trajectory_selection not in {"positive", "negative", "zero", "all", "zero_negative", "zero_positive"}:
        raise ValueError("opsd_trajectory_selection must be positive, negative, zero, all, zero_negative or zero_positive.")
    if not math.isfinite(config.opsd_distill_loss_coef) or config.opsd_distill_loss_coef < 0:
        raise ValueError("opsd_distill_loss_coef must be finite and non-negative.")
    if config.use_rubric_rlsd or config.whiten_advantages or config.add_token_level_kl:
        raise ValueError("GRPO+OPSD requires rubric-only advantages without RLSD, whitening or token KL.")
    if config.gamma != 1 or config.norm_mean_type != "group" or config.norm_std_type != "group":
        raise ValueError("GRPO+OPSD requires gamma=1 and group mean/std reward normalization.")
    if config.opsd_feedback_mode != "none":
        raise ValueError("GRPO+OPSD uses a fixed teacher prompt, without rollout feedback.")
    if config.saf_opd_topk_percent != 100 or config.saf_opd_tanh_scale is not None:
        raise ValueError("GRPO+OPSD does not combine SAF weighting with trajectory selection.")
    if config.force_disable_old_logprobs_recompute:
        raise ValueError("GRPO+OPSD requires old logprob recomputation.")
    threshold = getattr(config, "opsd_zero_min_unsatisfied_positive", 0)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("opsd_zero_min_unsatisfied_positive must be a non-negative integer.")
    if threshold and config.opsd_trajectory_selection not in {"zero", "zero_negative", "zero_positive"}:
        raise ValueError("Low-score zero filtering requires a selection containing zero.")
    score_max = getattr(config, "opsd_score_exclusive_max", None)
    if score_max is not None and (not math.isfinite(score_max) or not 0 <= score_max <= 1):
        raise ValueError("opsd_score_exclusive_max must be finite and in [0, 1], or None.")
    sigma = getattr(config, "opsd_score_gaussian_sigma", None)
    if sigma is not None:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("opsd_score_gaussian_sigma must be finite and positive, or None.")
        center = getattr(config, "opsd_score_gaussian_center", 0.5)
        if not math.isfinite(center) or not 0 <= center <= 1:
            raise ValueError("opsd_score_gaussian_center must be finite and in [0, 1].")
    weighting = getattr(config, "opsd_score_weighting", "symmetric")
    if weighting not in {"symmetric", "positive_lowzero_plateau"}:
        raise ValueError("Unknown opsd_score_weighting policy.")
    if weighting == "positive_lowzero_plateau":
        floor = getattr(config, "opsd_score_gaussian_floor", 0.1)
        if not math.isfinite(floor) or not 0 <= floor <= 1:
            raise ValueError("opsd_score_gaussian_floor must be finite and in [0, 1].")
        if sigma is None or config.loss_agg_mode != "token-mean":
            raise ValueError("Plateau OPSD requires Gaussian sigma and token-mean loss.")
        if config.opsd_trajectory_selection != "zero_positive" or threshold or score_max is not None:
            raise ValueError("Plateau OPSD requires zero_positive without extra score or criterion gates.")
    if getattr(config, "opsd_score_adaptive", False):
        if weighting != "positive_lowzero_plateau":
            raise ValueError("Adaptive scores require positive_lowzero_plateau weighting.")
        low, high = config.opsd_score_quantile_low, config.opsd_score_quantile_high
        if not 0 <= low < high <= 1:
            raise ValueError("Adaptive score quantiles require 0 <= low < high <= 1.")
        if not 0 < config.opsd_score_ema_rate <= 1:
            raise ValueError("Adaptive score EMA rate must be in (0, 1].")
        if not math.isfinite(config.opsd_score_sigma_min) or config.opsd_score_sigma_min <= 0:
            raise ValueError("Adaptive score sigma minimum must be finite and positive.")
    ablation = getattr(config, "opsd_score_ablation", "none")
    if ablation not in {"none", "unit", "uniform_budget"}:
        raise ValueError("Unknown OPSD score ablation.")
    if ablation != "none" and weighting != "positive_lowzero_plateau":
        raise ValueError("OPSD score ablations require plateau weighting.")
    negative_weight = getattr(config, "opsd_negative_weight", 0.0)
    if not math.isfinite(negative_weight) or not 0 <= negative_weight <= 1:
        raise ValueError("opsd_negative_weight must be finite and in [0, 1].")
    if negative_weight and (weighting != "positive_lowzero_plateau" or ablation != "unit"):
        raise ValueError("Weak negative teaching requires plateau unit weights.")
    validate_zero_gate(config)
    validate_teacher_refresh(config)
    validate_distill_switch(config)
    config.enable_old_logprobs_recompute = True


def validate_distill_switch(config: Any) -> None:
    """The switch replaces the OPSD weight after one rollout step; nothing else changes."""
    step = getattr(config, "opsd_distill_switch_step", None)
    after = getattr(config, "opsd_distill_loss_coef_after_switch", 0.0)
    if not math.isfinite(after) or after < 0:
        raise ValueError("opsd_distill_loss_coef_after_switch must be finite and non-negative.")
    auto = getattr(config, "opsd_distill_auto_switch", False)
    if auto:
        for key in ("opsd_auto_switch_window", "opsd_auto_switch_patience", "opsd_auto_switch_min_step"):
            value = getattr(config, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if key.endswith("min_step") else 1):
                raise ValueError(f"{key} must be a {'non-negative' if key.endswith('min_step') else 'positive'} integer.")
        tol = config.opsd_auto_switch_rel_tol
        if not math.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("opsd_auto_switch_rel_tol must be in (0, 1).")
    elif getattr(config, "opsd_auto_switch_history_seed", None):
        raise ValueError("opsd_auto_switch_history_seed requires opsd_distill_auto_switch.")
    if step is None:
        if after and not auto:
            raise ValueError("opsd_distill_loss_coef_after_switch requires opsd_distill_switch_step or auto switching.")
        return
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("opsd_distill_switch_step must be a non-negative integer or None.")


def effective_switch_step(config: Any, auto_step: Optional[int] = None) -> Optional[int]:
    """Last rollout step with the base weight: the earlier of the plateau trigger and the configured step."""
    deadline = getattr(config, "opsd_distill_switch_step", None)
    if auto_step is None:
        return deadline
    return auto_step if deadline is None else min(deadline, auto_step)


def scheduled_distill_coef(config: Any, global_step: int, auto_step: Optional[int] = None) -> float:
    """OPSD weight for this rollout step: the base weight, then the after-switch weight past the switch step."""
    step = effective_switch_step(config, auto_step)
    if step is not None and global_step > step:
        return config.opsd_distill_loss_coef_after_switch
    return config.opsd_distill_loss_coef


def distillation_skipped(config: Any, global_step: int, auto_step: Optional[int] = None) -> bool:
    """Only a scheduled switch to zero skips the teacher; a configured static alpha=0 keeps the full path."""
    return (effective_switch_step(config, auto_step) is not None
            and scheduled_distill_coef(config, global_step, auto_step) == 0)


def load_fkl_history_seed(path: Optional[str], before_step: int) -> List[List[float]]:
    """Plateau-test history of the steps before `before_step` from a JSON list of [step, selected-token FKL].

    Used when resuming a checkpoint written without the automatic switch, so the test sees the same history
    as an end-to-end run instead of starting empty.
    """
    if not path:
        return []
    history = [[int(step), float(value)] for step, value in json.loads(Path(path).read_text())
               if int(step) < before_step]
    steps = [step for step, _ in history]
    if steps != sorted(set(steps)) or not all(math.isfinite(v) and v >= 0 for _, v in history):
        raise ValueError(f"{path}: seed steps must be increasing and FKL values finite and non-negative.")
    return history


def fkl_plateau(history: Sequence[Sequence[float]], window: int, rel_tol: float, patience: int,
                min_step: int) -> Tuple[bool, Optional[float]]:
    """Plateau test on [step, selected-token FKL] history, one entry per distilled step.

    For each of the last `patience` entries, compare the mean FKL of its trailing window with the window
    before it; a relative decrease below `rel_tol` at all of them (and past `min_step`) is a plateau.
    Returns (plateau, latest relative decrease).
    """
    values = [float(v) for _, v in history]
    drops = []
    for end in range(len(values) - patience + 1, len(values) + 1):
        if end < 2 * window:
            return False, None
        previous = sum(values[end - 2 * window:end - window]) / window
        recent = sum(values[end - window:end]) / window
        drops.append((previous - recent) / previous if previous > 0 else 0.)
    latest = drops[-1] if drops else None
    ready = bool(history) and history[-1][0] + 1 >= min_step and len(drops) == patience
    return ready and all(d < rel_tol for d in drops), latest


def validate_zero_gate(config: Any) -> None:
    """Dropping the low-score gate only makes sense when plateau weighting distills zero advantages."""
    if getattr(config, "opsd_plateau_zero_score_gate", True):
        return
    if (getattr(config, "opsd_score_weighting", "symmetric") != "positive_lowzero_plateau"
            or not getattr(config, "opsd_plateau_include_zero", True)):
        raise ValueError("Ungated zero-advantage teaching requires plateau weighting with zero trajectories included.")


def validate_teacher_refresh(config: Any) -> None:
    """A refreshed privileged teacher is a Megatron snapshot of the student with the same layout."""
    step = getattr(config, "opsd_teacher_refresh_step", None)
    if step is None:
        return
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("opsd_teacher_refresh_step must be a non-negative integer or None.")
    teachers = list(getattr(config, "_reference_configs", {}).values())
    if len(teachers) != 1:
        raise ValueError("Teacher refresh supports exactly one OPSD teacher.")
    source, target = config.actor_train.strategy_args, teachers[0].strategy_args
    if source.strategy_name != "megatron_train" or target.strategy_name != "megatron_infer":
        raise ValueError("Teacher refresh requires Megatron train/infer strategies.")
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if source.strategy_config.get(key, 1) != target.strategy_config.get(key, 1):
            raise ValueError(f"Teacher refresh requires matching {key}.")
    if list(config.actor_train.device_mapping or []) != list(teachers[0].device_mapping or []):
        raise ValueError("Teacher refresh requires the teacher to share the student device mapping.")


def grpo_opsd_prediction_mask(
    advantages: torch.Tensor, response_mask: torch.Tensor, selection: str,
    unsatisfied_positive_count: torch.Tensor = None, zero_min_unsatisfied_positive: int = 0,
    rubric_scores: torch.Tensor = None, score_exclusive_max: float = None,
) -> torch.Tensor:
    """Select whole trajectories and align response tokens to predicting logits.

    GRPO advantages have one sign throughout a response. Zero-advantage responses
    are included in ``zero`` and ``all``. Selection never changes the GRPO mask.
    """
    valid = response_mask.bool()
    score = advantages.detach().float().masked_fill(~valid, 0).sum(-1)
    zero = (advantages.detach().float().abs().masked_fill(~valid, 0)
            .le(1e-6).all(-1) & valid.any(-1))
    eligible_zero = zero
    if zero_min_unsatisfied_positive:
        if unsatisfied_positive_count is None:
            raise ValueError("Low-score zero selection requires judge criterion counts.")
        counts = unsatisfied_positive_count.to(valid.device)
        if counts.shape != zero.shape:
            raise ValueError("Judge criterion counts must have one value per trajectory.")
        eligible_zero = zero & (counts >= zero_min_unsatisfied_positive)
    if selection == "positive":
        selected = score > 0
    elif selection == "negative":
        selected = score < 0
    elif selection == "zero":
        selected = eligible_zero
    elif selection == "zero_negative":
        selected = eligible_zero | ((score < 0) & ~zero)
    elif selection == "zero_positive":
        selected = eligible_zero | ((score > 0) & ~zero)
    elif selection == "all":
        selected = valid.any(-1)
    else:
        raise ValueError("opsd_trajectory_selection must be positive, negative, zero, all, zero_negative or zero_positive.")
    if score_exclusive_max is not None:
        if rubric_scores is None:
            raise ValueError("OPSD score filtering requires original rubric scores.")
        rubric_scores = rubric_scores.detach().to(valid.device)
        if rubric_scores.shape != selected.shape:
            raise ValueError("Rubric scores must have one value per trajectory.")
        # Apply after trajectory selection, so both positive and zero advantages
        # are filtered. Never alter rewards, advantages, or the GRPO loss mask.
        selected = selected & torch.isfinite(rubric_scores) & (rubric_scores < score_exclusive_max)
    mask = torch.zeros((valid.shape[0], valid.shape[1] + 1), dtype=torch.bool, device=valid.device)
    mask[:, :-1] = valid & selected.unsqueeze(-1)
    return mask


def unsatisfied_positive_count(weights, judgments, success=True):
    """Count missing positive-weight criteria; failed judgments cannot qualify."""
    if not success:
        return -1
    if judgments is None or len(weights) != len(judgments):
        raise ValueError("Successful judge result must contain one judgment per criterion.")
    return sum(float(weight) > 0 and not present for weight, present in zip(weights, judgments))


def gaussian_score_weights(scores: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
    """Detached per-trajectory weights; no batch or weight-sum normalization."""
    if scores.ndim != 1:
        raise ValueError("Gaussian OPSD weights require one original rubric score per trajectory.")
    scores = scores.detach().float()
    weights = torch.exp(-0.5 * ((scores - center) / sigma).square())
    return torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))


def plateau_gaussian_opsd_weights(
    scores: torch.Tensor, advantages: torch.Tensor, valid: torch.Tensor,
    judge_ok: torch.Tensor, lengths: torch.Tensor, response_limit: int,
    center: float, sigma: float, floor: float, include_zero: bool = True, zero_score_gate: bool = True,
) -> torch.Tensor:
    """Weight positive and low-score zero advantages; reject invalid/truncated rows.

    Positive weights are one below center, then decay towards floor. Zero
    advantages qualify strictly below center (or at any score without the score gate).
    Negative advantages receive zero.
    The caller retains all valid response tokens in the loss denominator.
    """
    if advantages.shape != valid.shape or scores.shape != (valid.shape[0],):
        raise ValueError("Plateau OPSD requires aligned scores, advantages and response masks.")
    if judge_ok is None or judge_ok.shape != scores.shape or lengths.shape != scores.shape:
        raise ValueError("Plateau OPSD requires per-response judge flags and lengths.")
    valid = valid.bool()
    scores = scores.detach().float().to(valid.device)
    adv = advantages.detach().float().masked_fill(~valid, 0)
    zero = adv.abs().le(1e-6).all(-1)
    positive = (adv.sum(-1) > 0) & ~zero
    quality = (valid.any(-1) & torch.isfinite(scores) & torch.isfinite(adv).all(-1)
               & (judge_ok.to(valid.device) == 1) & (lengths.to(valid.device) < response_limit))
    gaussian = floor + (1 - floor) * torch.exp(-0.5 * ((scores - center).clamp_min(0) / sigma).square())
    zero_selected = zero & include_zero & ((scores < center) if zero_score_gate else True)
    weights = torch.where(positive, gaussian, zero_selected.float())
    return torch.where(quality, weights, torch.zeros_like(weights))


def adaptive_score_parameters(
    scores: torch.Tensor, eligible: torch.Tensor, previous: dict | None,
    low: float, high: float, rate: float, sigma_min: float,
    fallback_center: float, fallback_sigma: float,
) -> tuple[float, float, dict | None, dict[str, float]]:
    """Compute rollout-wide quantiles and checkpointable EMA, before minibatch splitting.

    Initialize from the first nonempty rollout. Empty rollouts preserve state.
    Smooth the raw quantile span, then clamp only the applied width.
    """
    values = scores.detach().float()[eligible.bool() & torch.isfinite(scores)]
    state = previous
    metrics = {"actor/opsd_quantile_samples": float(values.numel())}
    if values.numel():
        qlow, qhigh = torch.quantile(values, values.new_tensor([low, high])).tolist()
        span = qhigh - qlow
        state = {"center": qlow, "span": span} if previous is None else {
            "center": (1 - rate) * previous["center"] + rate * qlow,
            "span": (1 - rate) * previous["span"] + rate * span,
        }
        metrics.update({"actor/opsd_score_q_low": qlow, "actor/opsd_score_q_high": qhigh})
    center = state["center"] if state is not None else fallback_center
    sigma = max(state["span"] if state is not None else fallback_sigma, sigma_min)
    metrics.update({"actor/opsd_score_center": center, "actor/opsd_score_sigma": sigma})
    return center, sigma, state, metrics


def ablate_score_weights(weights: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """Replace selected weights across the whole rollout, before minibatch splitting."""
    weights = weights.detach()
    selected = weights > 0
    if mode == "none":
        return weights
    if mode == "unit":
        return selected.to(weights.dtype)
    if mode == "uniform_budget":
        lengths = valid.bool().sum(-1).to(weights.dtype)
        mean = (weights * lengths).sum() / (selected * lengths).sum().clamp_min(1)
        return selected.to(weights.dtype) * mean
    raise ValueError("Unknown OPSD score ablation.")
