"""Full supported coordinator/driver with conformance container and fake provider."""
from dataclasses import replace
import tomllib
import pytest

from fakes import scenarios
from fakes.container import serve
from plane_harness import CanonicalClient, PlaneClock, config_text
from test_binder import FakeProvider
from test_plane import StubProvider, sampling_for

from synth_optimizers.rl.experiment import ExperimentSpec, ExperimentStore
from synth_optimizers.rl.experiment_driver import ContainerExperimentDriver
from synth_optimizers.rl.experiment_runner import run_experiment
from synth_optimizers.rl.plane import build_plane


class Provider(FakeProvider):
    tokenize_chat = StubProvider.tokenize_chat
    decode_tokens = StubProvider.decode_tokens


@pytest.mark.parametrize('interrupt', [False, True])
def test_full_driver_screen_exact_resume_select_and_final(tmp_path, interrupt):
    base = scenarios.multi_turn_environment_reward()
    container = serve(replace(base, advertised_concurrency=16, strong_task_digests=True,
        splits={'train': base.task_ids[:2], 'eval': base.task_ids[2:]}))
    provider = Provider()
    clock = PlaneClock(container)

    class Client(CanonicalClient):
        def rollout_state(self, rollout_id):
            clock.advance(0.1)
            return super().rollout_state(rollout_id)

    client = Client(container)
    requested_sampling = []
    def factory(config, *, sampling, admission_check):
        requested_sampling.append(sampling)
        # The conformance fake owns its fixed fingerprint; record the actual
        # driver's requested profile separately. Real adapters have own tests.
        return build_plane(config, client=client, provider=provider, clock=clock.run,
                           sampling=sampling_for(container), admission_check=admission_check)

    try:
        payload = tomllib.loads(config_text(container.config, container.base_url,
            groups_per_step=3, slots=12, max_open_groups=3, maximum_sampled_groups=12))
        payload['artifacts'] = {'directory': str(tmp_path/'artifacts'), 'catalog': str(tmp_path/'catalog.db')}
        payload['budget'] = {'experiment_id': 'offline', 'ledger': str(tmp_path/'budget.db'),
            'cap_usd': 10, 'input_usd_per_million': 1, 'output_usd_per_million': 1,
            'training_usd_per_million': 10000}
        rows = {}
        for split in ('train', 'eval'):
            rows[split] = [client.taskset_tasks({'split': split, 'ids': [task_id]})['rows'][0]
                           for task_id in container.config.splits[split]]
        def task(row):
            return {k: row[k] for k in ('task_id', 'seed', 'content_digest')}
        spec = ExperimentSpec.model_validate({'experiment_id': 'offline', 'run': payload,
            'train': [task(rows['train'][0])], 'validation': [task(rows['eval'][0])],
            'final': [task(rows['eval'][1])], 'updates': 2, 'segment_updates': 1,
            'validation_updates': [1, 2], 'evaluation_url': container.base_url,
            'screening': {'samples': 2, 'concurrency': 8},
            'judge_protocol_digest': 'sha256:'+'1'*64})
        store = ExperimentStore(tmp_path/'experiment.db')
        store.submit(spec)
        class Driver(ContainerExperimentDriver):
            def perform(self, phase, snapshot):
                result = super().perform(phase, snapshot)
                if interrupt and phase['id'] == 'train_1':
                    raise RuntimeError('injected crash after durable phase result, before coordinator commit')
                return result
        driver = Driver(spec, plane_factory=factory)
        if interrupt:
            from synth_optimizers.rl.experiment_service import ExperimentService
            with pytest.raises(RuntimeError, match='injected crash'):
                run_experiment(store, spec.experiment_id, driver)
            assert len(provider.train_calls) == 1
            service = ExperimentService(tmp_path/'experiment.db', driver_factory=lambda _: driver)
            service.control(spec.experiment_id, 'recover')
        result = run_experiment(store, spec.experiment_id, driver)
        assert result['state'] == 'completed'
        assert len(provider.train_calls) == 2
        assert len(provider.restore_calls) == 2
        assert all(c.kind == 'training_state' for c in provider.restore_calls)
        final = result['phases'][-1]['result']
        assert final['panel'] == 'final'
        assert final['paired_bootstrap_95_interval'] == [0, 0]
        assert [p.temperature for p in requested_sampling] == [1, 1, 1, 0, 0, 0]
        run_experiment(ExperimentStore(tmp_path/'experiment.db'), spec.experiment_id, driver)
        assert len(provider.train_calls) == 2  # completed work is never replayed
        from synth_optimizers.rl.experiment_service import ExperimentService
        service = ExperimentService(tmp_path/'experiment.db')
        cursor, events = 0, []
        while True:
            page = service.events(spec.experiment_id, cursor, limit=10)
            events.extend(page['events'])
            cursor = page['next_sequence']
            if not page['has_more']:
                break
        assert any(e['event_type'].startswith('runtime.') for e in events)
        assert any(e['event_type'] == 'evaluation.attempt_completed' for e in events)
        assert len({e['event_id'] for e in events}) == len(events)
    finally:
        container.shutdown()
