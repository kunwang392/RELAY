from functools import partial

import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.pipeline.distill.various_divergence import VariousDivergence
from roll.utils.opsd_metrics import distribution_diagnostics, agreement_diagnostics
from roll.utils.relay_only import saf_distribution_weights
from roll.utils.relay_only import seed_opd_terms
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl
from roll.utils.train_infer_corrections import compute_train_infer_correction

class ActorWorker(BaseActorWorker):

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        if self.pipeline_config.uses_opsd_seed_loss:
            return self._compute_opsd_seed_loss(data, output_tensor)
        if (self.pipeline_config.uses_opsd_distribution_loss
                and not getattr(self.pipeline_config, "use_grpo_opsd", False)):
            return self._compute_opsd_distribution_loss(data, output_tensor)

        aggregate, approx_kl = agg_loss, compute_approx_kl
        use_ruscarl = getattr(self.pipeline_config, "use_ruscarl", False)
        if use_ruscarl:
            from roll.utils.relay_only import policy_logits, microbatch_agg_loss, reference_approx_kl
            output_tensor = policy_logits(output_tensor, self.pipeline_config)
            aggregate = partial(microbatch_agg_loss, loss_scale=data.meta_info["loss_scale"])
            approx_kl = reference_approx_kl

        response_mask = data.batch["response_mask"][:, 1:].long()
        final_response_mask = data.batch.get("final_response_mask", response_mask)
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]
        cripo_loss, cripo_metrics = None, {}
        if getattr(self.pipeline_config, "use_cripo", False):
            from megatron.core import mpu
            from roll.utils.relay_only import cripo_actor_terms
            advantages, cripo_loss, cripo_metrics = cripo_actor_terms(
                data, output_tensor, self.pipeline_config, mpu.get_tensor_model_parallel_group(),
            )
            data.meta_info.pop("cripo_teacher_output", None)

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']
        if 'final_response_mask' not in batch_num_tokens:
            batch_num_tokens['final_response_mask'] = batch_num_tokens['response_mask']
            global_valid_samples['final_response_mask'] = global_valid_samples['response_mask']

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)
        infer_log_probs = infer_log_probs if len(infer_log_probs) > 0 else old_log_probs

        train_infer_metric = {}
        if not self.pipeline_config.enable_old_logprobs_recompute:
            train_infer_is_weight, filter_mask, train_infer_metric = compute_train_infer_correction(
                cfg=self.pipeline_config.train_infer_correction,
                response_mask=response_mask,
                old_log_probs=old_log_probs,
                infer_log_probs=infer_log_probs,
                global_valid_samples=global_valid_samples['response_mask'],
                global_valid_tokens=batch_num_tokens['response_mask'],
            )

            # Apply filter mask to both response_mask and final_response_mask
            response_mask = response_mask.long() * filter_mask.long()
            final_response_mask = final_response_mask.long() * filter_mask.long()
        else:
            train_infer_is_weight = data.batch['train_infer_is_weight']

        valid_samples = torch.any(final_response_mask > 0, dim=1).float()
        sample_weights = self.compute_sample_weights(data, response_mask)

        kl_loss = approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=final_response_mask, kl_penalty="k3"
        )
        kl_loss = aggregate(loss_mat=kl_loss,
                        loss_mask=final_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                        batch_num_tokens=batch_num_tokens['final_response_mask'],
                        global_valid_samples=global_valid_samples['final_response_mask'],)

        approxkl = approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        if self.pipeline_config.importance_sampling == "token":
            log_ratio = log_probs - old_log_probs
            ratio = (log_ratio.clamp(-20, 20) if use_ruscarl else log_ratio).exp()
        elif self.pipeline_config.importance_sampling == "seq":
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)
            ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)

        pg_clip_low = self.pipeline_config.pg_clip_low if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages

        loss = -torch.min(surr1, surr2)

        if self.pipeline_config.dual_clip_loss:
            clip_c = self.pipeline_config.ruscarl_clip_ratio_c if use_ruscarl else (1 + self.pipeline_config.pg_clip * 2)
            dual_clip_loss = -torch.max(-loss, clip_c * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        if self.pipeline_config.train_infer_correction.is_weight.enabled:
            loss = loss * train_infer_is_weight

        if getattr(self.pipeline_config, "grpo_skip_negative_pg", False):
            # GRPO has a constant trajectory advantage. Keep the denominator and
            # original advantages/masks intact for OPSD selection and reference KL.
            negative = ((advantages.detach().float().masked_fill(~final_response_mask.bool(), 0).sum(-1) < 0)
                        & ~(advantages.detach().float().abs().masked_fill(~final_response_mask.bool(), 0)
                            .le(1e-6).all(-1)))
            loss = loss.masked_fill(negative.unsqueeze(-1), 0)

        weighted_pg_loss = aggregate(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    weights=sample_weights,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)
        original_pg_loss = aggregate(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)

        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = weighted_pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = weighted_pg_loss

        total_loss = total_loss * self.pipeline_config.rl_loss_coef

        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = aggregate(
                loss_mat=entropy,
                loss_mask=data.batch["response_mask"][:, 1:],
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        loss_metric = {
            "actor/ppo_ratio_high_clipfrac@sum": aggregate(loss_mat=clipped_high, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_low_clipfrac@sum": aggregate(loss_mat=clipped_low, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_clipfrac@sum": aggregate(loss_mat=clipped, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ratio_mean@sum": aggregate(loss_mat=ratio, loss_mask=response_mask,
                                loss_agg_mode='seq-mean-token-mean',
                                 global_valid_samples=global_valid_samples['response_mask']).detach().item(),
            "actor/ratio_max@max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min@min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac@sum": aggregate(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, batch_num_tokens=batch_num_tokens['final_response_mask'],
                                       global_valid_samples=global_valid_samples['response_mask']).detach().item(),
        }

        pg_metrics = {
            "actor/pg_loss@sum": original_pg_loss.detach().item(),
            "actor/weighted_pg_loss@sum": weighted_pg_loss.detach().item(),
            "actor/kl_loss@sum": kl_loss.detach().item(),
            "actor/total_loss@sum": total_loss.detach().item(),
            "actor/approxkl@sum": aggregate(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens['response_mask'],
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/policykl@sum": aggregate(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens['response_mask'],
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/valid_samples@sum": valid_samples.sum().detach().item(),
            "actor/total_samples@sum": float(valid_samples.size(0)),
            "actor/valid_sample_ratio@sum": (valid_samples.sum() / global_valid_samples['response_mask']).detach().item(),
            "actor/sample_weights_mean@mean": sample_weights.mean().detach().item(),
            "actor/sample_weights_min@min": sample_weights.min().detach().item(),
            "actor/sample_weights_max@max": sample_weights.max().detach().item(),
            **loss_metric,
            **train_infer_metric,
        }

        # A scheduled switch to zero weight leaves no teacher outputs in the batch.
        if (getattr(self.pipeline_config, "use_grpo_opsd", False)
                and not data.meta_info.get("opsd_distill_skipped", False)):
            distill_loss, distill_metrics = self._compute_opsd_distribution_loss(data, output_tensor)
            total_loss = total_loss + distill_loss
            pg_metrics.update(distill_metrics)
            pg_metrics["actor/opsd_weighted_loss@sum"] = distill_loss.detach().item()
            pg_metrics["actor/total_loss@sum"] = total_loss.detach().item()

        if cripo_loss is not None:
            total_loss = total_loss + cripo_loss
            pg_metrics.update(cripo_metrics)
            pg_metrics["actor/total_loss@sum"] = total_loss.detach().item()
        return total_loss, pg_metrics

    def _compute_opsd_seed_loss(self, data: DataProto, output_tensor: torch.Tensor):
        teacher_name = next(iter(self.pipeline_config.reference_configs))
        mask = data.batch.get("final_response_mask", data.batch["response_mask"][:, 1:]).bool()
        student = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"],
            attention_mask=data.batch["response_mask"],
        )
        terms, gate, delta = seed_opd_terms(
            student, data.batch[f"ref_log_probs_{teacher_name}"], mask,
            beta=self.pipeline_config.opsd_seed_beta,
        )
        counts = data.meta_info["batch_num_tokens"]
        count = counts.get("final_response_mask", counts["response_mask"])

        def token_mean(values):
            return agg_loss(values.masked_fill(~mask, 0), mask, "token-mean", batch_num_tokens=count)

        loss = token_mean(terms)
        total_loss = self.pipeline_config.opsd_distill_loss_coef * loss
        metrics = {
            "actor/opsd_distill_loss@sum": loss.detach().item(),
            "actor/total_loss@sum": total_loss.detach().item(),
        }
        # Keep all SEED diagnostics in one tracker group, with global token weighting.
        with torch.no_grad():
            for name, values in {
                "gate_mean": gate,
                "delta_mean": delta,
                "delta_abs_mean": delta.abs(),
                "teacher_supported_fraction": (delta > 0).float(),
                "gate_below_01_fraction": (gate < 0.1).float(),
                "gate_above_09_fraction": (gate > 0.9).float(),
                "student_nll": -student,
            }.items():
                metrics[f"seed_opd/{name}@sum"] = token_mean(values).item()
        return total_loss, metrics

    def _compute_opsd_distribution_loss(self, data: DataProto, output_tensor: torch.Tensor):
        teacher_name = next(iter(self.pipeline_config.reference_configs))
        teacher_logits = data.batch[f"opsd_teacher_topk_logits_{teacher_name}"].detach()
        teacher_indices = data.batch[f"opsd_teacher_topk_indices_{teacher_name}"].detach()
        prediction_mask = data.batch["opsd_prediction_mask"]

        scaled_teacher_logits = teacher_logits / self.pipeline_config.opsd_temperature
        teacher_inf_mask = scaled_teacher_logits.isinf()
        teacher_log_probs = torch.log_softmax(scaled_teacher_logits, dim=-1, dtype=torch.float32)
        teacher_probs = teacher_log_probs.exp()
        divergence = VariousDivergence(self.pipeline_config)
        diagnostics = {}
        loss_mat, _ = self.strategy.op_compute_various_divergence(
            loss_callable=divergence,
            logits=output_tensor,
            teacher_topk_probs=teacher_probs,
            teacher_topk_log_probs=teacher_log_probs,
            teacher_topk_indices=teacher_indices,
            teacher_topk_inf_mask=teacher_inf_mask,
            labels=None,
            attention_mask=prediction_mask,
            reduction="none",
            diagnostics_fn=partial(distribution_diagnostics, temperature=self.pipeline_config.opsd_temperature),
            diagnostics=diagnostics,
        )
        loss_mat = loss_mat * prediction_mask
        fkl_mat = loss_mat.detach()  # before trajectory weights: the auto-switch plateau signal

        score_sigma = getattr(self.pipeline_config, "opsd_score_gaussian_sigma", None)
        trajectory_weights = data.batch.get("opsd_trajectory_weights")
        score_weighting_enabled = score_sigma is not None or trajectory_weights is not None
        if trajectory_weights is not None:
            score_weights = trajectory_weights.detach().to(loss_mat.device)
            if score_weights.shape != (loss_mat.shape[0],):
                raise ValueError("OPSD weights must match the trajectory batch size.")
            score_unweighted_loss_mat = loss_mat.detach()
            loss_mat = loss_mat * score_weights.unsqueeze(-1)
        elif score_sigma is not None:
            from roll.utils.grpo_opsd import gaussian_score_weights

            if "scores" not in data.batch:
                raise ValueError("Gaussian OPSD weighting requires original rubric scores.")
            score_weights = gaussian_score_weights(
                data.batch["scores"],
                getattr(self.pipeline_config, "opsd_score_gaussian_center", 0.5), score_sigma,
            ).to(loss_mat.device)
            if score_weights.shape != (loss_mat.shape[0],):
                raise ValueError("Gaussian OPSD weights must match the trajectory batch size.")
            score_unweighted_loss_mat = loss_mat.detach()
            # Keep the existing selected-token/sample denominators. Dividing by
            # sum(weights) would cancel the intended attenuation in low-weight batches.
            loss_mat = loss_mat * score_weights.unsqueeze(-1)

        saf_percent = getattr(self.pipeline_config, "saf_opd_topk_percent", 100.0)
        saf_scale = getattr(self.pipeline_config, "saf_opd_tanh_scale", None)
        saf_enabled = saf_percent != 100 or saf_scale is not None
        if saf_enabled:
            unweighted_loss_mat = loss_mat.detach()
            with torch.no_grad():
                student_sampled_logp = self.strategy.op_compute_log_probs(
                    logits=output_tensor.detach(), input_ids=data.batch["input_ids"],
                    attention_mask=data.batch["response_mask"],
                )
                teacher_sampled_logp = data.batch[f"ref_log_probs_{teacher_name}"].detach()
                if teacher_sampled_logp.shape != student_sampled_logp.shape:
                    raise ValueError("SAF teacher/student sampled logprob shapes differ.")
                if student_sampled_logp.shape != prediction_mask[:, :-1].shape:
                    raise ValueError("SAF sampled logprobs must align with next-token prediction positions.")
                # logprob column t scores input_ids[t+1], so it belongs to the
                # loss/logits at t. CP strategies have already gathered the full sequence.
                saf_raw = torch.zeros_like(loss_mat, dtype=torch.float32)
                saf_raw[:, :-1] = teacher_sampled_logp.float() - student_sampled_logp.float()
                saf_raw.masked_fill_(~prediction_mask.bool(), 0)
                saf_weights, saf_processed, saf_gate = saf_distribution_weights(
                    saf_raw, prediction_mask, saf_percent, saf_scale,
                )
            loss_mat = loss_mat * saf_weights

        # A whole batch (or just a DP microbatch) may contain no selected trajectories.
        # Probe cohorts share an all-valid-token denominator so rare cohorts are not amplified.
        normalization_key = (
            "opsd_normalization_mask" if "opsd_normalization_mask" in data.batch else
            "opsd_probe_normalization_mask" if "opsd_probe_normalization_mask" in data.batch
            else "opsd_prediction_mask"
        )
        batch_num_tokens = max(data.meta_info["batch_num_tokens"][normalization_key], 1)
        distill_loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=prediction_mask,
            loss_agg_mode=self.pipeline_config.loss_agg_mode,
            batch_num_tokens=batch_num_tokens,
            global_valid_samples=data.meta_info.get("global_valid_samples", {}).get(normalization_key),
        )
        # GRPO+OPSD passes the scheduled weight per step; other modes use the configured weight.
        coef = data.meta_info.get("opsd_distill_loss_coef", self.pipeline_config.opsd_distill_loss_coef)
        total_loss = distill_loss * coef
        distill_loss_value = distill_loss.detach().item()
        selected_tokens = max(data.meta_info["batch_num_tokens"].get("opsd_prediction_mask", 0), 1)
        metrics = {
            "actor/opsd_distill_loss@sum": distill_loss_value,
            "actor/total_loss@sum": distill_loss_value * coef,
            # Mean FKL per selected token, independent of the loss denominator.
            "actor/opsd_selected_token_fkl@sum": agg_loss(
                loss_mat=fkl_mat, loss_mask=prediction_mask, loss_agg_mode="token-mean",
                batch_num_tokens=selected_tokens,
            ).item(),
        }
        # These describe the RENORMALIZED teacher top-k distribution, not the
        # full-vocabulary entropy or probability mass. No additional forward pass.
        with torch.no_grad():
            def token_mean(values):
                return agg_loss(
                    loss_mat=values.masked_fill(~prediction_mask.bool(), 0), loss_mask=prediction_mask,
                    loss_agg_mode="token-mean", batch_num_tokens=batch_num_tokens,
                ).item()

            if score_weighting_enabled:
                mean_weight = token_mean(score_weights.unsqueeze(-1).expand_as(prediction_mask))
                metrics.update({
                    "actor/opsd_score_weight_mean@sum": mean_weight,
                    "actor/opsd_score_effective_coef@sum": mean_weight * coef,
                    "actor/opsd_score_unweighted_loss@sum": token_mean(score_unweighted_loss_mat),
                })

            if saf_enabled:
                metrics.update({
                    "actor/saf_opd_keep_fraction@sum": token_mean(saf_gate.float()),
                    "actor/saf_opd_weight_mean@sum": token_mean(saf_weights),
                    "actor/saf_opd_raw_abs_mean@sum": token_mean(saf_raw.abs()),
                    "actor/saf_opd_processed_abs_mean@sum": token_mean(saf_processed.abs()),
                    "actor/saf_opd_unweighted_loss@sum": token_mean(unweighted_loss_mat),
                })
                if prediction_mask.bool().any():
                    metrics["actor/saf_opd_raw_abs_max@max"] = saf_raw.abs()[prediction_mask.bool()].max().item()
                    metrics["actor/saf_opd_processed_abs_max@max"] = saf_processed.abs()[prediction_mask.bool()].max().item()

            student_values, student_indices = self.strategy.op_compute_topk_logits(
                output_tensor.detach(), topk=teacher_indices.shape[-1]
            )
            agreement = agreement_diagnostics(student_values, student_indices, teacher_logits, teacher_indices)
            for name, values in agreement.items():
                metrics[f"actor/{name}@sum"] = token_mean(values)
            for name, values in diagnostics.items():
                if name == "cv_max":
                    valid = values[prediction_mask.bool()]
                    # Empty microbatches must not bias max reduction toward zero.
                    if valid.numel():
                        metrics["actor/cv_max@max"] = valid.max().item()
                else:
                    metrics[f"actor/{name}@sum"] = token_mean(values)

            entropy = -torch.where(
                teacher_probs > 0, teacher_probs * teacher_log_probs, 0.0
            ).sum(-1)
            top_two = teacher_probs.topk(2, dim=-1).values
            valid_losses = loss_mat.detach()[prediction_mask.bool()]
            metrics.update({
                "actor/teacher_topk_entropy@sum": token_mean(entropy),
                "actor/teacher_topk_top1_probability@sum": token_mean(top_two[..., 0]),
                "actor/teacher_topk_top1_top2_gap@sum": token_mean(top_two[..., 0] - top_two[..., 1]),
                "actor/opsd_token_loss_max@max": valid_losses.max().item() if valid_losses.numel() else 0.0,
                "actor/opsd_token_loss_min@min": valid_losses.min().item() if valid_losses.numel() else 0.0,
                "actor/opsd_nonzero_loss_token_fraction@sum": token_mean((loss_mat.detach().abs() > 1e-8).float()),
                # Zero only when no token of the step is distilled; the plateau test then skips the step.
                "actor/opsd_selected_token_fraction@sum": token_mean(prediction_mask.float()),
            })
        return total_loss, metrics

    def compute_sample_weights(self, data: DataProto, response_mask: torch.Tensor):
        """
        可以基于难度和长度的样本权重
        """
        batch_size = response_mask.shape[0]
        sample_weights = torch.ones(batch_size, device=response_mask.device)

        # 1. 基于难度的权重 - 例如：难度越高，权重越大
        if self.pipeline_config.difficulty_loss_weight and "difficulty" in data.non_tensor_batch:
            try:
                difficulty = data.non_tensor_batch["difficulty"]
                if isinstance(difficulty, np.ndarray):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                elif not isinstance(difficulty, torch.Tensor):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                norm_difficulty = torch.clamp(difficulty, 0.0, 1.0)
                difficulty_weights = 0.5 + 1.5 * norm_difficulty
                sample_weights = sample_weights * difficulty_weights
            except Exception as e:
                self.logger.warning(f"跳过difficulty权重计算：{str(e)}")

        # 2. 基于长度的权重 - 例如：长度越长，权重越小
        response_lengths = response_mask.sum(dim=1).float()
        if self.pipeline_config.length_loss_weight:
            # 同样归一化长度到[0.5, 2.0]范围
            norm_lengths = (response_lengths - response_lengths.min()) / (
                    response_lengths.max() - response_lengths.min() + 1e-8
            )
            length_weights = 1.5 - norm_lengths
            sample_weights = sample_weights * length_weights

        if sample_weights.sum() > 0:
            sample_weights = sample_weights * (batch_size / (sample_weights.sum() + 1e-8))

        return sample_weights
