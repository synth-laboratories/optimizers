import pytest

from synth_optimizers.cispo_service import CispoService, CispoServiceError
from synth_optimizers.rl.experiment_service import ExperimentService
from test_experiment import spec


def test_public_service_routes_same_identity_and_cursor(tmp_path):
    experiments = ExperimentService(tmp_path/'experiments.db')
    service = CispoService(tmp_path/'legacy.db', fixture=True, experiments=experiments)
    design = spec(tmp_path)
    try:
        result = service.submit(design.model_dump(), run_id=design.experiment_id)
        assert result['run_id'] == design.experiment_id
        assert result['status'] == 'ready'
        page = service.optimizer_events(design.experiment_id)
        assert page['schema_version'] == 'optimizer_event_page.v1'
        assert page['events'][0]['sequence_number'] == 1
        assert service.optimizer_events(design.experiment_id) == page
        service.experiment_control(design.experiment_id, 'pause')
        assert service.get(design.experiment_id)['status'] == 'paused'
        assert service.cancel(design.experiment_id)['status'] == 'stopped'
        with pytest.raises(CispoServiceError):
            service.submit(design.model_dump(), run_id='different')
    finally:
        service.store.close()


def test_outbox_import_is_idempotent_and_budget_does_not_reset(tmp_path):
    from synth_optimizers.rl.budget import ExperimentBudget
    service = ExperimentService(tmp_path/'experiments.db')
    design = spec(tmp_path)
    service.submit(design.model_dump())
    budget = ExperimentBudget(tmp_path/'budget.db', 'exp-a', 10)
    budget.reserve('sample-1', 'sampling', 1)
    first = service.events('exp-a')
    assert any(e['event_type'].startswith('budget.') for e in first['events'])
    service = ExperimentService(tmp_path/'experiments.db')
    assert service.events('exp-a') == first
    assert service.get('exp-a')['budget']['counted_or_reserved_usd'] == 1


def test_recover_never_claims_new_work(tmp_path):
    from synth_optimizers.rl.experiment import CoordinationError
    service = ExperimentService(tmp_path/'experiments.db')
    service.submit(spec(tmp_path).model_dump())
    with pytest.raises(CoordinationError):
        service.control('exp-a', 'recover')
    assert service.get('exp-a')['status'] == 'ready'


def test_stop_does_not_hide_events_from_a_draining_phase(tmp_path):
    service = ExperimentService(tmp_path/'experiments.db')
    service.submit(spec(tmp_path).model_dump())
    claim = service.store.claim('exp-a')
    service.control('exp-a', 'stop')
    assert service.get('exp-a')['status'] == 'stopping'
    assert not service.events('exp-a')['terminal']
    service.store.complete('exp-a', claim, {})
    assert service.get('exp-a')['status'] == 'stopped'
    assert service.events('exp-a')['terminal']


def test_authenticated_http_experiment_contract(tmp_path):
    import json
    import threading
    import urllib.request
    import urllib.error
    from synth_optimizers.cispo_service import create_cispo_http_server
    experiments = ExperimentService(tmp_path/'experiments.db')
    service = CispoService(tmp_path/'legacy.db', fixture=True, experiments=experiments)
    with pytest.raises(CispoServiceError, match='bearer token'):
        create_cispo_http_server(('127.0.0.1', 0), service)
    server = create_cispo_http_server(('127.0.0.1', 0), service, service_token='offline-test-token')
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    def request(path, payload=None):
        req = urllib.request.Request(origin+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization': 'Bearer offline-test-token', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            return json.load(response)
    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(origin+'/v1/capabilities')
        assert missing.value.code == 401
        assert request('/v1/capabilities')['container_experiments']
        assert request('/v1/runs', {'run_id': 'exp-a', 'config_json': spec(tmp_path).model_dump()})['status'] == 'ready'
        assert request('/v1/runs/exp-a/optimizer-events')['events'][0]['event_type'] == 'experiment.prepared'
        assert request('/v1/runs/exp-a/evaluations')['panels'] == []
        assert request('/v1/runs/exp-a/state/batch?slices=checkpoints')['checkpoints']['items'] == []
        assert request('/v1/runs/exp-a/stop', {})['status'] == 'stopped'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        service.store.close()
