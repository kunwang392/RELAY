"""Reject excluded methods while retaining the shared ROLL runtime interfaces."""

def _unsupported(*args, **kwargs):
    raise ValueError("This minimal release includes RELAY only; the requested optional method is not included.")


def validate_cripo(config):
    if getattr(config, "use_cripo", False):
        _unsupported()


def validate_ruscarl(config):
    if getattr(config, "use_ruscarl", False):
        _unsupported()


def validate_saf_options(percent, scale):
    if percent != 100 or scale is not None:
        _unsupported()

SgLangStrategy = _unsupported

_CRITERION_HEADER = _unsupported

apply_rubric_rlsd = _unsupported

balance_cripo_batch = _unsupported

create_sampling_params_for_sglang = _unsupported

cripo_actor_terms = _unsupported

microbatch_agg_loss = _unsupported

parse_support = _unsupported

partition_positive_criteria = _unsupported

policy_logits = _unsupported

postprocess_generate = _unsupported

prepare_cripo = _unsupported

prepare_scaffold_requests = _unsupported

privileged_criteria = _unsupported

privileged_criteria_items = _unsupported

reference_approx_kl = _unsupported

reference_response_mask = _unsupported

render_feedback_messages = _unsupported

response_sentences = _unsupported

restore_original_prompt = _unsupported

rubric_metadata = _unsupported

saf_distribution_weights = _unsupported

saf_opd_advantages = _unsupported

scaffold_group_metrics = _unsupported

seed_opd_terms = _unsupported

support_request = _unsupported

teacher_forward = _unsupported
