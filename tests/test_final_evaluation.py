"""Synthetic integration checks; no datasets, candidate models or live judge requests."""
import ast
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roll.evaluation import final_eval as ev

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = '1. [POSITIVE REQUIREMENT, weight=+5] Requirement.\n2. [NEGATIVE PENALTY, weight=-2] Penalty.'


def rubric_row():
    return {'id': 'q1', 'messages': [{'role': 'user', 'content': 'Question'}], 'prompt_str': 'Question',
            'rubrics_str': RUBRICS, 'rubric_weights': [5, -2], 'rubric_count': 2}


def turns_row():
    return {'id': 'conversation', 'llmeval_category': '医疗知识', 'llmeval_turns': [
        {'question': 'First question', 'reference': 'Private reference', 'checklist': ''},
        {'question': 'Follow-up', 'reference': 'Another private reference', 'checklist': ''}]}


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, seeds):
        self.calls.append((copy.deepcopy(messages), list(seeds)))
        return [{'text': f'Candidate {seed}', 'tokens': 3, 'finish_reason': 'stop'} for seed in seeds]


class FakeClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **payload):
        self.calls.append(payload)
        if self.fail:
            raise RuntimeError('synthetic private credential must never be persisted')
        properties = payload['response_format']['json_schema']['schema']['properties']
        content = ({'score': '[4]', 'explanation': 'Checked'} if 'score' in properties else
                   {key: 'PRESENT' for key in properties})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                                                       message=SimpleNamespace(content=json.dumps(content)))], usage=None)


def record(benchmark='rar_medicine', repeat=0):
    row = turns_row() if benchmark == 'llmeval_med' else rubric_row()
    return next(ev.generate_records(benchmark, [row], FakeGenerator(), repeat))


def test_generation_keeps_full_candidate_history_and_uses_fresh_repeat_seeds():
    rows = ev.validate_rows('llmeval_med', [turns_row()], check_size=False)
    generator = FakeGenerator()
    first = list(ev.generate_records('llmeval_med', rows, generator, 0))
    second = list(ev.generate_records('llmeval_med', rows, generator, 1))
    history = generator.calls[1][0][0]
    assert history[2] == {'role': 'assistant', 'content': first[0]['answers'][0]['text']}
    assert 'Private reference' not in json.dumps(generator.calls)
    assert first[0]['answers'][0]['seed'] != second[0]['answers'][0]['seed']
    assert first == list(ev.generate_records('llmeval_med', rows, FakeGenerator(), 0))
    assert ev.SAMPLING['max_tokens'] == 5120 and ev.SAMPLING['temperature'] == .7


def test_generator_enforces_5120_context_budget_and_non_thinking(monkeypatch):
    import sys
    calls = []
    tokens = list(range(12))
    def template(messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return tokens
    def generate(prompts, sampling_params, use_tqdm):
        calls.append((prompts, sampling_params))
        assert sampling_params[0].max_tokens == 5120 and sampling_params[0].seed == 7
        return [SimpleNamespace(outputs=[SimpleNamespace(text='Answer', token_ids=[1, 2], finish_reason='stop')])]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    generator = ev.VLLMGenerator.__new__(ev.VLLMGenerator)
    generator.engine = SimpleNamespace(generate=generate)
    generator.tokenizer = SimpleNamespace(apply_chat_template=template)
    generator.max_model_len = 5132
    assert generator([[{'role': 'user', 'content': 'Q'}]], [7])[0]['text'] == 'Answer'
    generator.max_model_len = 5131
    with pytest.raises(ValueError, match='never silently truncated'):
        generator([[{'role': 'user', 'content': 'Q'}]], [7])
    assert len(calls) == 1


def test_signed_judgment_schema_and_healthbench_aggregation():
    client = FakeClient()
    graded = asyncio.run(ev.GPTJudge(client).grade(record()))
    payload = client.calls[0]
    assert payload['model'] == 'gpt-5.1-2025-11-13' and payload['reasoning_effort'] == 'none'
    assert payload['response_format']['json_schema']['strict'] is True
    assert 'temperature' not in payload and 'extra_body' not in payload
    assert 'NEGATIVE PENALTY' in payload['messages'][0]['content']
    assert graded['grades'][0]['grade']['score'] == pytest.approx(.6)
    health = [dict(grades=[dict(success=True, grade=dict(score=-.5))]),
              dict(grades=[dict(success=True, grade=dict(score=1.))])]
    assert ev.summarize_grades('healthbench', health)['score'] == 25


def test_llmeval_judge_uses_prior_candidate_history_and_turn_weighted_mean():
    sample = record('llmeval_med')
    requests = ev.judge_requests(sample)
    assert sample['answers'][0]['text'] in requests[1][0]
    assert 'Private reference' not in requests[1][0]
    client = FakeClient()
    grades = asyncio.run(ev.GPTJudge(client).grade(sample))
    assert len(grades['grades']) == 2
    assert ev.summarize_grades('llmeval_med', [grades])['score'] == 80
    mixed = [{'grades': [{'success': True, 'grade': {'score': x}} for x in [1, 1]]},
             {'grades': [{'success': True, 'grade': {'score': 4}}]}]
    assert ev.summarize_grades('llmeval_med', mixed)['score'] == 40


def test_failures_do_not_become_scores_and_error_text_is_not_saved():
    result = asyncio.run(ev.GPTJudge(FakeClient(fail=True), attempts=1).grade(record()))
    assert result['success'] is False
    assert 'private credential' not in json.dumps(result)
    summary = ev.summarize_grades('rar_medicine', [result])
    assert summary['score'] is None and summary['failed_judgments'] == 1


def test_retry_preserves_successful_turn_judgments():
    sample = record('llmeval_med')
    previous = asyncio.run(ev.GPTJudge(FakeClient()).grade(sample))
    first = copy.deepcopy(previous['grades'][0])
    previous['grades'][1] = {'success': False}
    client = FakeClient()
    retried = asyncio.run(ev.GPTJudge(client).grade(sample, previous))
    assert len(client.calls) == 1 and retried['grades'][0] == first


def test_external_input_validation_requires_full_splits_and_unit_researchqa():
    with pytest.raises(ValueError, match='expected 2242'):
        ev.validate_rows('rar_medicine', [rubric_row()])
    with pytest.raises(ValueError, match='duplicate'):
        ev.validate_rows('rar_medicine', [rubric_row(), rubric_row()], check_size=False)
    with pytest.raises(ValueError, match='unit weights'):
        ev.validate_rows('researchqa', [rubric_row()], check_size=False)
    source = rubric_row()
    ev.validate_rows('rar_medicine', [source], check_size=False)
    assert source == rubric_row()


def test_three_repeat_directory_can_resume_without_rejudging_or_resampling(tmp_path):
    manifest = {'domain': 'science', 'model_size': '1.7b', 'protocol': ev.protocol_settings(),
                'inputs': {'rar_science': {'rows': 1}, 'researchqa': {'rows': 1}}}
    ev.write_json(tmp_path / 'manifest.json', manifest)
    for repeat in range(3):
        for benchmark in manifest['inputs']:
            sample = record(benchmark, repeat)
            path = tmp_path / f'repeat-{repeat + 1}-{benchmark}.responses.jsonl'
            path.write_text(json.dumps(sample) + '\n')
    client = FakeClient()
    summary = asyncio.run(ev.judge_directory(tmp_path, client=client))
    assert len(client.calls) == 6 and summary['complete']
    assert summary['benchmarks']['rar_science']['mean'] == pytest.approx(60)
    assert summary['benchmarks']['rar_science']['std'] == 0
    again = asyncio.run(ev.judge_directory(tmp_path, client=client))
    assert len(client.calls) == 6 and again == summary
    path = tmp_path / 'repeat-1-rar_science.responses.jsonl'
    tampered = json.loads(path.read_text())
    tampered['answers'][0]['text'] = 'changed answer'
    path.write_text(json.dumps(tampered) + '\n')
    with pytest.raises(ValueError, match='Saved answers changed'):
        asyncio.run(ev.judge_directory(tmp_path, client=client))


def test_aggregate_is_unweighted_over_five_benchmarks_and_checks_domains():
    summaries = [{'domain': domain, 'model_size': '4b', 'protocol': ev.protocol_settings(), 'complete': True,
                  'benchmarks': {n: {'mean': 10 if domain == 'medicine' else 60}
                                 for n in names}} for domain, names in ev.DOMAIN_BENCHMARKS.items()]
    assert ev.combine_domains(summaries)['aggregate'] == 30
    summaries[1]['model_size'] = '1.7b'
    with pytest.raises(ValueError, match='same model size'):
        ev.combine_domains(summaries)


def final_validation_method():
    path = ROOT / 'roll/pipeline/rlvr/rlvr_pipeline.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'RLVRPipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_final_policy')
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    return ns['validate_final_policy']


@pytest.mark.parametrize('checkpoint_step,expected', [(399, [400]), (140, [])])
def test_final_validation_syncs_final_checkpoint_before_step_400(checkpoint_step, expected):
    events, logged = [], []
    def val(global_step):
        assert events[:2] == ['offload_train', ('sync', 400)]
        assert 'load_infer' in events and 'load_reward' in events
        events.append(('validate', global_step))
        return {'val/score': .6}
    pipeline = SimpleNamespace(state=SimpleNamespace(step=checkpoint_step), val_dataset=[1],
        pipeline_config=SimpleNamespace(max_steps=400),
        actor_train=SimpleNamespace(offload_states=lambda **_: events.append('offload_train')),
        model_update=lambda step: events.append(('sync', step)),
        actor_infer=SimpleNamespace(load_states=lambda **_: events.append('load_infer'),
                                   offload_states=lambda **_: events.append('offload_infer')),
        rewards={'judge': SimpleNamespace(load_states=lambda: events.append('load_reward'),
                                          offload_states=lambda: events.append('offload_reward'))},
        val=val, tracker=SimpleNamespace(log=lambda values, step: logged.append((step, values))))
    final_validation_method()(pipeline)
    assert [s for s, _ in logged] == expected
    assert pipeline.state.step == checkpoint_step
    if expected:
        assert logged[0][1]['validation/checkpoint_step'] == 399
        assert events[-2:] == ['offload_infer', 'offload_reward']
