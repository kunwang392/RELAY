import math

import torch
import torch.nn as nn

from roll.pipeline.distill.distill_config import DistillConfig

from roll.utils.constants import IGNORE_INDEX

class VariousDivergence:
    def __init__(self, pipeline_config: DistillConfig, padding_id=IGNORE_INDEX) -> None:
        self.kd_temperature = getattr(
            pipeline_config,
            "opsd_temperature",
            getattr(pipeline_config, "kd_temperature", 1.0),
        )
        self.kd_objective = getattr(
            pipeline_config,
            "opsd_loss_type",
            getattr(pipeline_config, "kd_objective", "forward_kl"),
        )
        self.jsd_beta = getattr(pipeline_config, "opsd_jsd_beta", 0.5)
        self.pointwise_clip = getattr(pipeline_config, "opsd_token_clip", None)
        self.padding_id = padding_id
        self.args = pipeline_config

        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")

    def __call__(self, logits, teacher_probs, teacher_log_probs, teacher_inf_mask):
        kld = self.dist_func(logits, teacher_probs, teacher_log_probs, teacher_inf_mask)
        return kld

    def _reduce_vocab_contributions(self, contributions, logits, teacher_inf_mask):
        inf_mask = logits.isinf() | teacher_inf_mask
        contributions.masked_fill_(inf_mask, 0.0)
        if self.pointwise_clip is not None and self.pointwise_clip > 0:
            contributions.clamp_(max=self.pointwise_clip)
        return contributions.sum(-1)

    def compute_forward_kl_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        logits = logits / self.kd_temperature
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)

        contributions = teacher_probs * (teacher_log_probs - lprobs)
        return self._reduce_vocab_contributions(
            contributions,
            logits,
            teacher_inf_mask,
        )

    def compute_reverse_kl_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        logits = logits / self.kd_temperature

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        contributions = probs * (lprobs - teacher_log_probs)
        return self._reduce_vocab_contributions(
            contributions,
            logits,
            teacher_inf_mask,
        )

    def compute_adaptive_kl_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        alpha = self.args.adaptive_kl_alpha
        probs = torch.softmax(
            logits / self.kd_temperature, dim=-1, dtype=torch.float32
        )
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_probs, teacher_log_probs, teacher_inf_mask)
        rkl = self.compute_reverse_kl_divergence(logits, teacher_probs, teacher_log_probs, teacher_inf_mask)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl

        return akl

    def compute_skewed_forward_kl_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        logits = logits / self.kd_temperature

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        kld = (teacher_probs * (teacher_log_probs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_inf_mask
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        return kld

    def compute_skewed_reverse_kl_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        logits = logits / self.kd_temperature

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_inf_mask
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        return kld

    def compute_js_divergence(
        self,
        logits,
        teacher_probs,
        teacher_log_probs,
        teacher_inf_mask,
    ):
        if self.jsd_beta == 0:
            return self.compute_forward_kl_divergence(
                logits,
                teacher_probs,
                teacher_log_probs,
                teacher_inf_mask,
            )
        if self.jsd_beta == 1:
            return self.compute_reverse_kl_divergence(
                logits,
                teacher_probs,
                teacher_log_probs,
                teacher_inf_mask,
            )

        logits = logits / self.kd_temperature
        student_log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        student_probs = student_log_probs.exp()
        mixture_log_probs = torch.logaddexp(
            student_log_probs + math.log1p(-self.jsd_beta),
            teacher_log_probs + math.log(self.jsd_beta),
        )
        contributions = (
            self.jsd_beta * teacher_probs * (teacher_log_probs - mixture_log_probs)
            + (1 - self.jsd_beta) * student_probs * (student_log_probs - mixture_log_probs)
        )
        return self._reduce_vocab_contributions(
            contributions,
            logits,
            teacher_inf_mask,
        )