"""Installed adapter startup and protocol gates; no credentials or paid calls."""
import hashlib
import json

import pytest

from synth_optimizers.rl.benchmark_server import create_benchmark_app
from synth_optimizers.rl.experiment import ExperimentSpec
from test_experiment import spec as base_spec


def design(tmp_path, benchmark, protocol):
    payload = base_spec(tmp_path).model_dump()
    payload.update(benchmark=benchmark, judge_protocol=protocol,
        judge_protocol_digest='sha256:'+hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        renderer_profile={'profile_id': 'renderers.fixture.v1', 'package': 'renderers',
            'package_version': '0.1.11', 'config_digest': 'sha256:'+'a'*64,
            'tokenizer_id': 'openai/gpt-oss-20b', 'tokenizer_digest': 'sha256:'+'b'*64,
            'stop_token_ids': [99]})
    payload['run']['plan']['groups_per_step'] = 3
    payload['run']['pipeline'] = {'max_execution_slots': 8}
    return payload


def test_installed_craftax_server_has_frozen_manifest_and_closes(tmp_path, monkeypatch):
    pytest.importorskip('craftax_gold')
    from craftax_gold import cispo
    from unittest.mock import Mock
    declaration = Mock(wraps=cispo.craftax_cispo_declaration)
    monkeypatch.setattr(cispo, 'craftax_cispo_declaration', declaration)
    monkeypatch.setenv('SYNTH_CRAFTAX_CISPO_MAX_CALLS', '32')
    from fastapi.testclient import TestClient
    protocol = {'adapter': 'craftax.environment_return.v1', 'env_steps': 200, 'normalization': 'none'}
    spec = ExperimentSpec.model_validate(design(tmp_path, 'craftax', protocol))
    app = create_benchmark_app(spec, temperature=0, engine_url='http://127.0.0.1:9')
    assert declaration.call_args.kwargs['policy_calls'] == 8
    assert cispo.task_rows(split_seeds={'train': (196001,), 'heldout': (197001,)}, env_step_limit=64)[0].content_digest != cispo.task_rows(split_seeds={'train': (196001,), 'heldout': (197001,)}, env_step_limit=120)[0].content_digest
    with TestClient(app) as client:
        from synth_optimizers.rl.contract import ContainerContract
        ContainerContract.from_metadata(client.get('/metadata').json())
        manifest = client.get('/rl/experiment').json()
        assert manifest['temperature'] == 0
        assert manifest['normalization'] == 'none'
        assert manifest['spec_digest'] == hashlib.sha256(spec.model_dump_json().encode()).hexdigest()


def test_installed_healthbench_protocol_and_text_dataset(tmp_path):
    cispo = pytest.importorskip('healthbench_chat.cispo')
    from fastapi.testclient import TestClient
    corpus = [{'prompt_id': str(i), 'prompt': [{'role': 'user', 'content': '  EXACT\nText  '}],
               'rubrics': [{'criterion': 'mentions care', 'points': 1}]} for i in range(3)]
    dataset = tmp_path/'dataset.jsonl'
    dataset.write_text('\n'.join(json.dumps(row) for row in corpus))
    protocol = {'identity': cispo.ProviderRubricJudge().identity(), 'temperature': 0, 'max_tokens': 512,
                'normalization': 'none', 'adapter': 'healthbench.rubric.v1'}
    payload = design(tmp_path, 'healthbench', protocol)
    payload.update(judge_input_usd_per_million=1, judge_output_usd_per_million=1)
    for split, task in zip(('train', 'validation', 'final'), cispo.declared_tasks(source=lambda: corpus, count=3)):
        payload[split] = [{'task_id': task.task_id, 'seed': task.seed, 'content_digest': task.content_digest}]
    spec = ExperimentSpec.model_validate(payload)
    app = create_benchmark_app(spec, temperature=1, dataset_path=dataset)
    with TestClient(app) as client:
        assert client.get('/rl/experiment').json()['budgeted_grading'] is True
        from synth_optimizers.rl.contract import ContainerContract
        ContainerContract.from_metadata(client.get('/metadata').json())
        response = client.post('/cispo/taskset/tasks', json={'ids': [t.task_id for t in spec.train], 'split': 'eval'})
        assert response.status_code == 200
        assert [r['task_id'] for r in response.json()['rows']] == [t.task_id for t in spec.train]
    payload['judge_protocol']['temperature'] = 1
    payload['judge_protocol_digest'] = 'sha256:'+hashlib.sha256(json.dumps(payload['judge_protocol'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='judge differs'):
        create_benchmark_app(ExperimentSpec.model_validate(payload), temperature=1, dataset_path=dataset)
