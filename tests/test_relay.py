"""CPU checks using synthetic tensors only; no datasets, judge calls, or GPU jobs."""
import ast
import asyncio
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roll.utils import grpo_opsd as core
from roll.utils.evaluation_protocol import parse_judgments, score_judgments
from roll.utils.rubric_training import group_minibatch_permutation
from roll.utils.relay_only import validate_cripo, validate_ruscarl, validate_saf_options


def rule(values, **overrides):
    params = dict(window=20, rel_tol=.05, patience=5, min_step=60)
    params.update(overrides)
    history = []
    for step, value in enumerate(values):
        history.append([step, value])
        if core.fkl_plateau(history, **params)[0]:
            return step
    return None


def config(**overrides):
    values = dict(use_grpo_opsd=True, opsd_distill_loss_coef=.01,
                  opsd_distill_switch_step=199, opsd_distill_loss_coef_after_switch=0.,
                  opsd_distill_auto_switch=True, opsd_auto_switch_window=20,
                  opsd_auto_switch_rel_tol=.05, opsd_auto_switch_patience=5,
                  opsd_auto_switch_min_step=60)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_plateau_minimum_history_and_step_boundary():
    assert rule([.2] * 100) == 59
    assert rule([.2] * 100, min_step=0) == 43
    assert rule([.6 * .995 ** t for t in range(400)]) is None
    assert 100 < rule([.18 + .4 * math.exp(-t / 35) for t in range(300)]) < 160


def test_plateau_uses_strict_threshold_and_handles_zero_previous_mean():
    h = [[i, v] for i, v in enumerate([1.] * 20 + [.5] * 20)]
    assert core.fkl_plateau(h, 20, .5, 1, 0) == (False, .5)
    assert core.fkl_plateau([[i, 0.] for i in range(44)], 20, .05, 5, 0) == (True, 0.)


@pytest.mark.parametrize("changes", [dict(opsd_auto_switch_window=0),
    dict(opsd_auto_switch_patience=0), dict(opsd_auto_switch_rel_tol=0),
    dict(opsd_auto_switch_rel_tol=1), dict(opsd_auto_switch_min_step=-1)])
def test_invalid_rule_settings_fail(changes):
    with pytest.raises(ValueError):
        core.validate_distill_switch(config(**changes))


def test_selection_excludes_high_ties_negative_failed_and_truncated_responses():
    scores = torch.tensor([.9, .2, .3, .1, .2, .2, float('nan')])
    adv = torch.tensor([[1., 1.], [0., 0.], [0., 0.], [-1., -1.], [1., 1.], [1., 1.], [0., 0.]])
    valid = torch.ones_like(adv, dtype=torch.bool)
    judge = torch.tensor([1, 1, 1, 1, 0, 1, 1])
    lengths = torch.tensor([2, 2, 2, 2, 2, 4, 2])
    weights = core.plateau_gaussian_opsd_weights(scores, adv, valid, judge, lengths, 4,
                                               center=.3, sigma=.25, floor=.1)
    selected = core.ablate_score_weights(weights, valid, "unit")
    assert selected.tolist() == [1., 1., 0., 0., 0., 0., 0.]
    assert scores[:6].tolist() == pytest.approx([.9, .2, .3, .1, .2, .2])


def test_prediction_mask_shifts_response_positions_without_mutating_grpo_mask():
    response = torch.tensor([[0, 1, 1], [0, 1, 0], [0, 1, 1]])
    advantages = torch.tensor([[0., 1., 1.], [0., 0., 0.], [0., -1., -1.]])
    before = response.clone()
    mask = core.grpo_opsd_prediction_mask(advantages, response, "zero_positive")
    assert mask.tolist() == [[False, True, True, False], [False, True, False, False], [False] * 4]
    assert torch.equal(before, response)


def test_threshold_initializes_updates_and_preserves_state_for_empty_batch():
    scores = torch.tensor([0., .2, .4, .6, .8, 1.])
    eligible = torch.ones(6, dtype=torch.bool)
    c, sigma, state, _ = core.adaptive_score_parameters(scores, eligible, None, .3, .7, .1, .05, .3, .25)
    assert c == pytest.approx(.3)
    assert sigma == pytest.approx(.4)
    c2, _, state2, _ = core.adaptive_score_parameters(scores + .1, eligible, state, .3, .7, .1, .05, .3, .25)
    assert c2 == pytest.approx(.31)
    c3, _, state3, _ = core.adaptive_score_parameters(scores, ~eligible, state2, .3, .7, .1, .05, .3, .25)
    assert state3 == state2 and c3 == c2


def pipeline_methods():
    path = ROOT / 'roll/pipeline/rlvr/rlvr_pipeline.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLVRPipeline')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_apply_grpo_opsd_schedule', '_update_opsd_auto_switch'}]
    ns = dict(DataProto=object, logger=SimpleNamespace(info=lambda *_: None),
              scheduled_distill_coef=core.scheduled_distill_coef,
              distillation_skipped=core.distillation_skipped, fkl_plateau=core.fkl_plateau,
              load_fkl_history_seed=core.load_fkl_history_seed)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def test_actual_pipeline_switches_once_and_skips_next_teacher_pass():
    ns = pipeline_methods()
    pipeline = SimpleNamespace(pipeline_config=config(), state=SimpleNamespace(kv={}))
    skipped_steps = []
    for step in range(100):
        batch = SimpleNamespace(meta_info={})
        manager = SimpleNamespace(metrics={'actor/opsd_selected_token_fkl@sum': .2}, add_metrics=lambda _: None)
        if ns['_apply_grpo_opsd_schedule'](pipeline, batch, step, manager):
            skipped_steps.append(step)
            assert batch.meta_info['opsd_distill_loss_coef'] == 0
        else:
            ns['_update_opsd_auto_switch'](pipeline, step, manager)
    assert skipped_steps == list(range(60, 100))
    assert pipeline.state.kv['opsd_auto_switch_step'] == 59
    assert len(pipeline.state.kv['opsd_auto_switch_history']) == 60
    restored = json.loads(json.dumps(pipeline.state.kv))
    assert core.distillation_skipped(config(), 100, restored['opsd_auto_switch_step'])


def test_empty_selected_batch_does_not_advance_history():
    ns = pipeline_methods()
    pipeline = SimpleNamespace(pipeline_config=config(), state=SimpleNamespace(kv={}))
    manager = SimpleNamespace(metrics={'actor/opsd_selected_token_fkl@sum': 0.,
                                      'actor/opsd_selected_token_fraction@sum': 0.}, add_metrics=lambda _: None)
    ns['_update_opsd_auto_switch'](pipeline, 90, manager)
    assert pipeline.state.kv == {}


def test_resume_seed_and_deadline(tmp_path):
    seed = tmp_path / 'history.json'
    seed.write_text(json.dumps([[i, .2] for i in range(60)]))
    assert len(core.load_fkl_history_seed(str(seed), 40)) == 40
    assert core.effective_switch_step(config(), 250) == 199
    assert core.scheduled_distill_coef(config(), 199) == .01
    assert core.scheduled_distill_coef(config(), 200) == 0
    seed.write_text('[[2, 0.2], [1, 0.1]]')
    with pytest.raises(ValueError):
        core.load_fkl_history_seed(str(seed), 40)


def test_actual_forward_kl_matches_conditional_distribution_and_has_gradient():
    path = ROOT / 'roll/pipeline/distill/various_divergence.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    ns = dict(torch=torch, nn=torch.nn, math=math, DistillConfig=object, IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    loss = ns['VariousDivergence'](SimpleNamespace(opsd_temperature=1., opsd_loss_type='forward_kl', opsd_token_clip=None))
    student = torch.tensor([[1., 2., 3.]], requires_grad=True)
    teacher_log = torch.log_softmax(torch.tensor([[3., 1., 2.]]), -1)
    teacher = teacher_log.exp()
    actual = loss.compute_forward_kl_divergence(student, teacher, teacher_log, torch.zeros_like(student, dtype=torch.bool))
    expected = (teacher * (teacher_log - torch.log_softmax(student, -1))).sum(-1)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_actual_reward_worker_passes_signed_training_rubrics_to_judge():
    path = ROOT / 'roll/pipeline/rlvr/rewards/rubric_evaluator_reward_worker.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RubricEvaluatorRewardWorker')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'compute_rewards')
    method.decorator_list = []
    ns = dict(torch=torch, asyncio=asyncio, RubricRLSDRewardWorker=type('Excluded', (), {}),
              OPSDFeedbackRewardWorker=type('ExcludedFeedback', (), {}),
              DataProto=SimpleNamespace(from_dict=lambda tensors: SimpleNamespace(batch=tensors, non_tensor_batch={})))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    rubrics = '1. [POSITIVE REQUIREMENT, weight=+5] First.\n2. [NEGATIVE PENALTY, weight=-2] Penalty.'
    async def evaluate(**kwargs):
        assert kwargs['rubrics_str'] == rubrics and kwargs['weights'] == [5, -2]
        assert kwargs['rubric_count'] == 2 and kwargs['temperature'] == .1
        metrics = score_judgments([True, True], kwargs['weights'])
        return SimpleNamespace(**metrics, success=True, attempts=1, latency=0., judgments=[True, True],
                               prompt_tokens=0, completion_tokens=0, rate_limited_count=0, server_error_count=0,
                               early_stop=False, whitespace_abort_count=0, usage_available=False,
                               format_error_count=0, truncated_count=0)
    worker = SimpleNamespace(judge_tags={'rubric_opsd_train'}, _empty_evaluation=lambda: None,
                             pipeline_config=SimpleNamespace(rubric_train_positive_only=False,
                                                             rubric_train_judge_temperature=.1),
                             actor_tokenizer=SimpleNamespace(batch_decode=lambda *a, **kw: ['candidate']),
                             _evaluate=evaluate, skip_failed_training_judge=True)
    data = SimpleNamespace(batch={'responses': torch.tensor([[1, 2]])}, non_tensor_batch={
        'tag': ['rubric_opsd_train'], 'rubrics_str': [rubrics], 'rubric_weights': [[5, -2]],
        'rubric_count': [2], 'prompt_str': ['Question']})
    result = asyncio.run(ns['compute_rewards'](worker, data))
    assert result.batch['scores'].item() == pytest.approx(.6)
    assert result.batch['rubric_negative_trigger_count'].item() == 1
    assert not result.batch['rubric_training_judge_failed'].item()
    assert data.non_tensor_batch['rubrics_str'] == [rubrics]
    worker.pipeline_config.rubric_train_positive_only = True
    with pytest.raises(ValueError, match='full signed'):
        asyncio.run(ns['compute_rewards'](worker, data))


def test_judge_parser_requires_complete_result_and_signed_scoring():
    result = parse_judgments('{"1":"PRESENT","2":"PRESENT","3":"NOT_PRESENT"}', 3)
    assert result == [True, True, False]
    assert score_judgments(result, [5, -2, 3])['score'] == pytest.approx(3 / 8)
    with pytest.raises(ValueError):
        parse_judgments('{"1":"PRESENT"}', 3)


def test_group_balancing_keeps_optimizer_minibatches_complete():
    lengths = torch.tensor([8, 1, 4, 2, 7, 3, 6, 5])
    order = group_minibatch_permutation(lengths, dp_size=2, minibatch_size=4)
    assert sorted(order.tolist()) == list(range(8))
    shards = order.reshape(2, 4)
    assert set(shards[:, :2].flatten().tolist()) == set(range(4))
    assert set(shards[:, 2:].flatten().tolist()) == set(range(4, 8))


@pytest.mark.parametrize('model', ['1.7b', '4b'])
@pytest.mark.parametrize('domain', ['medicine', 'science'])
def test_four_launch_profiles_resolve_without_accessing_data(model, domain, monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_API_KEY', 'synthetic-secret-for-test')
    monkeypatch.setenv('PRETRAIN_MODEL', 'must-not-override-explicit-model')
    result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--model', model,
                             '--domain', domain, '--print-config'], capture_output=True, text=True, check=True)
    assert 'synthetic-secret-for-test' not in result.stdout
    cfg = OmegaConf.create(result.stdout)
    assert cfg.pretrain == ('Qwen/Qwen3-4B' if model == '4b' else 'Qwen/Qwen3-1.7B')
    assert cfg.opsd_privileged_context_key == 'rubrics_str'
    assert cfg.opsd_auto_switch_window == 20 and cfg.opsd_auto_switch_patience == 5
    assert cfg.student_train.training_args.weight_decay == 0
    assert cfg.max_steps == 400 and cfg.num_gpus_per_node == 8
    assert cfg.rubric_train_positive_only is False
    assert cfg.validation.generating_args.max_new_tokens == cfg.response_length == 5120
    assert cfg.val_sequence_length - cfg.val_prompt_length >= 5120


def test_excluded_features_fail_explicitly():
    validate_cripo(SimpleNamespace(use_cripo=False))
    validate_ruscarl(SimpleNamespace(use_ruscarl=False))
    validate_saf_options(100, None)
    with pytest.raises(ValueError):
        validate_cripo(SimpleNamespace(use_cripo=True))
    with pytest.raises(ValueError):
        validate_saf_options(50, None)


def test_launcher_rejects_positive_only_rewards_and_effective_length_overrides():
    spec = importlib.util.spec_from_file_location('relay_launcher', ROOT / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = OmegaConf.load(ROOT / 'configs/relay.yaml')
    module.validate_method(cfg)
    for key, value in [('rubric_train_positive_only', True), ('val_sequence_length', 16384),
                       ('val_sequence_length', 20000), ('response_length', 4096)]:
        altered = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        altered[key] = value
        with pytest.raises(ValueError):
            module.validate_method(altered)


def test_local_tracker_writes_metric_values_without_cloud_dependencies(tmp_path):
    from roll.utils.tracking import create_tracker
    tracker = create_tracker('jsonl', {'seed': 42}, log_dir=str(tmp_path), name='run')
    tracker.log({'actor/loss@sum': torch.tensor(.25)}, step=7)
    tracker.finish()
    assert json.loads((tmp_path / 'run/config.json').read_text()) == {'seed': 42}
    assert json.loads((tmp_path / 'run/metrics.jsonl').read_text()) == {'step': 7, 'actor/loss': .25}
