"""Detached OPSD diagnostics; all distribution outputs are per sequence position."""

import torch


@torch.no_grad()
def distribution_diagnostics(logits, teacher_probs, teacher_log_probs, teacher_inf_mask, temperature=1.0):
    """Forward KL on teacher support, before clipping, and its individual contributions."""
    student_lp = torch.log_softmax(logits / temperature, dim=-1, dtype=torch.float32)
    contributions = teacher_probs * (teacher_log_probs - student_lp)
    contributions = contributions.masked_fill(logits.isinf() | teacher_inf_mask, 0)
    student_probs = student_lp.exp()
    return {
        "forward_kl_topk": contributions.sum(-1),
        "cv_max": contributions.max(-1).values,
        "cv_abs_mean": contributions.abs().mean(-1),
        "student_teacher_support_entropy": -torch.where(
            student_probs > 0, student_probs * student_lp, 0.0
        ).sum(-1),
    }


@torch.no_grad()
def agreement_diagnostics(student_values, student_indices, teacher_values, teacher_indices):
    """Compare independently selected global-vocabulary top-k sets (not restricted argmax)."""
    student_top1 = student_indices.gather(-1, student_values.argmax(-1, keepdim=True)).squeeze(-1)
    teacher_top1 = teacher_indices.gather(-1, teacher_values.argmax(-1, keepdim=True)).squeeze(-1)
    # Search sorted IDs to avoid materializing a [B, S, K, K] comparison tensor.
    ordered = teacher_indices.long().sort(-1).values.contiguous()
    indices = student_indices.long().contiguous()
    positions = torch.searchsorted(ordered, indices).clamp_max(ordered.shape[-1] - 1)
    matches = ordered.gather(-1, positions).eq(indices)
    return {
        "top1_agreement": student_top1.eq(teacher_top1).float(),
        "topk_overlap": matches.float().mean(-1),
        "student_top1_in_teacher_topk": teacher_indices.eq(student_top1.unsqueeze(-1)).any(-1).float(),
    }


def response_repetition_metrics(input_ids, response_mask, special_token_ids=()):
    """Macro average per-response token-ID repetition, excluding prompts/padding/special IDs.

    A response shorter than n has repetition 0. Report eligible fraction to make
    that convention visible. Repeated occurrences after the first count as repeats.
    """
    rows = input_ids.detach().cpu().tolist()
    masks = response_mask.detach().cpu().bool().tolist()
    special = set(special_token_ids)
    responses = [[t for t, valid in zip(row, mask) if valid and t not in special]
                 for row, mask in zip(rows, masks)]
    metrics = {}
    for n in (3, 4):
        rates = []
        eligible = 0
        for tokens in responses:
            total = len(tokens) - n + 1
            if total <= 0:
                rates.append(0.0)
                continue
            eligible += 1
            unique = len({tuple(tokens[i:i+n]) for i in range(total)})
            rates.append(1.0 - unique / total)
        prefix = f"response/repetition_{n}gram"
        metrics[f"{prefix}/mean"] = sum(rates) / len(rates) if rates else 0.0
        metrics[f"{prefix}/max"] = max(rates, default=0.0)
        metrics[f"{prefix}/eligible_fraction"] = eligible / len(rates) if rates else 0.0
    metrics["response/empty_fraction"] = sum(not r for r in responses) / len(responses) if responses else 0.0
    return metrics
