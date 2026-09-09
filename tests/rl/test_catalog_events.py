"""Checkpoint stream crash consistency and public read semantics; no paid calls."""
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from synth_optimizers.rl.catalog import CatalogError, CheckpointCatalog
from synth_optimizers.rl.read_api import capabilities, checkpoint_details, run_snapshot
from test_catalog import make_record


def test_registration_publication_reconnect_and_isolation(tmp_path):
    path = tmp_path / 'catalog.db'
    with CheckpointCatalog(path) as catalog:
        record = make_record(checkpoint_id='a')
        catalog.register_checkpoint(record)
        catalog.register_checkpoint(record)
        catalog.register_checkpoint(make_record(checkpoint_id='b', run_id='other'))
        catalog.record_publication('a', 'published')
        first = catalog.event_page('run_a', limit=1)
        assert first['has_more']
        assert first == catalog.event_page('run_a', limit=1)
    with CheckpointCatalog(path) as catalog:
        rest = catalog.event_page('run_a', after_sequence=first['next_sequence'])
        assert rest['log_id'] == first['log_id']
        assert [r['sequence_number'] for r in rest['events']] == [2, 3]
        assert rest['events'][-1]['fields']['publication_status'] == 'published'
        assert len(catalog.event_page('other')['events']) == 2
        assert catalog.event_page('run_a', after_sequence=3)['next_sequence'] == 3


def test_event_and_state_rollback_together(tmp_path):
    with CheckpointCatalog(tmp_path / 'catalog.db') as catalog:
        with pytest.raises(RuntimeError):
            with catalog.transaction():
                catalog.register_checkpoint(make_record(checkpoint_id='a'))
                raise RuntimeError('crash before commit')
        assert not catalog.has_checkpoint('a')
        assert not catalog.event_page('run_a')['events']
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        assert catalog.event_page('run_a')['events'][0]['sequence_number'] == 1


def test_outbox_failure_prevents_source_commit(tmp_path):
    with CheckpointCatalog(tmp_path / 'catalog.db') as catalog:
        catalog._conn.execute("CREATE TRIGGER fail_event BEFORE INSERT ON checkpoint_event_outbox "
                              "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match='injected'):
            catalog.register_checkpoint(make_record(checkpoint_id='a'))
        assert not catalog.has_checkpoint('a')


def test_effective_view_does_not_claim_verified_resume(tmp_path):
    with CheckpointCatalog(tmp_path / 'catalog.db') as catalog:
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        catalog.record_publication('a', 'published')
        catalog.put_alias('selected', 'checkpoint', 'a')
        result = checkpoint_details(catalog, 'a')
        assert result['checkpoint']['publication_status'] == 'published'
        assert catalog.get_checkpoint('a').publication_status == 'staged'
        assert result['aliases'][0]['alias'] == 'selected'
        assert result['resume']['has_training_state']
        assert not result['resume']['eligible']
        assert result['artifact_health']['status'] == 'unverified'


@pytest.mark.parametrize('kwargs', [{'after_sequence': -1}, {'limit': 0}, {'limit': 2001},
                                   {'after_sequence': True}])
def test_invalid_cursor_or_page_refused(tmp_path, kwargs):
    with CheckpointCatalog(tmp_path / 'catalog.db') as catalog:
        with pytest.raises(CatalogError):
            catalog.event_page('run_a', **kwargs)


def test_concurrent_connections_assign_unique_run_sequence(tmp_path):
    path = tmp_path / 'catalog.db'
    with CheckpointCatalog(path):
        pass
    def write(i):
        with CheckpointCatalog(path) as catalog:
            catalog.register_checkpoint(make_record(checkpoint_id=f'c{i}', update_id=f'u{i}'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    with CheckpointCatalog(path) as catalog:
        events = catalog.event_page('run_a')['events']
        assert [r['sequence_number'] for r in events] == list(range(1, 25))
        assert len({r['event_id'] for r in events}) == 24


def test_capabilities_are_honest():
    assert capabilities()['checkpoint_events']
    assert not capabilities()['remote_controls']
    assert not capabilities()['historical_event_backfill']


def test_alias_and_availability_are_audited_with_state(tmp_path):
    with CheckpointCatalog(tmp_path/'catalog.db') as catalog:
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        catalog.put_alias('selected', 'checkpoint', 'a')
        catalog.record_artifact_observation('a', {'artifacts': {'training_state': {'available': False}}})
        kinds = [event['event_type'] for event in catalog.event_page('run_a')['events']]
        assert kinds[-2:] == ['checkpoint.alias_changed', 'checkpoint.availability_checked']
        assert checkpoint_details(catalog, 'a')['alias_history']
        for table in ('checkpoint_artifact_observations', 'checkpoint_alias_history'):
            with pytest.raises(sqlite3.IntegrityError):
                catalog._conn.execute(f'DELETE FROM {table}')


def test_public_cli_uses_same_projection(tmp_path, capsys):
    from synth_optimizers.cli import build_parser
    from synth_optimizers.rl.cli import dispatch

    path = tmp_path / 'catalog.db'
    with CheckpointCatalog(path) as catalog:
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        expected = checkpoint_details(catalog, 'a')
    args = build_parser().parse_args(['rl', 'catalog', 'details', '--catalog', str(path), 'a'])
    assert dispatch(args) == 0
    assert json.loads(capsys.readouterr().out) == expected
    args = build_parser().parse_args(['rl', 'catalog', 'events', '--catalog', str(path),
                                     '--run', 'run_a', '--limit', '1'])
    assert dispatch(args) == 0
    assert len(json.loads(capsys.readouterr().out)['events']) == 1


def test_catalogs_with_reused_run_ids_have_distinct_log_identity(tmp_path):
    with CheckpointCatalog(tmp_path / 'a.db') as a, CheckpointCatalog(tmp_path / 'b.db') as b:
        assert a.event_page('run_a')['log_id'] != b.event_page('run_a')['log_id']


def test_migration_does_not_invent_historical_events(tmp_path):
    path = tmp_path / 'catalog.db'
    with CheckpointCatalog(path) as catalog:
        for table in ('checkpoints', 'publication_events', 'save_attempts'):
            catalog._conn.execute(f'DROP TRIGGER {table}_checkpoint_event_v1')
        catalog.register_checkpoint(make_record(checkpoint_id='historical'))
    with CheckpointCatalog(path) as catalog:
        assert checkpoint_details(catalog, 'historical')['checkpoint']['checkpoint_id'] == 'historical'
        assert not catalog.event_page('run_a')['events']
        catalog.record_publication('historical', 'published')
        assert len(catalog.event_page('run_a')['events']) == 1


def test_failed_publication_rolls_back_event_and_view(tmp_path):
    with CheckpointCatalog(tmp_path / 'catalog.db') as catalog:
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        before = catalog.event_page('run_a')
        with pytest.raises(RuntimeError):
            with catalog.transaction():
                catalog.record_publication('a', 'published')
                raise RuntimeError('publication interrupted')
        assert catalog.event_page('run_a') == before
        assert checkpoint_details(catalog, 'a')['checkpoint']['publication_status'] == 'staged'


def test_snapshot_cursor_and_future_publication(tmp_path):
    path = tmp_path / 'catalog.db'
    with CheckpointCatalog(path) as catalog:
        catalog.register_checkpoint(make_record(checkpoint_id='a'))
        snapshot = run_snapshot(catalog, 'run_a')
        assert snapshot['cursor']['after_sequence'] == 2
        assert snapshot['checkpoints'][0]['checkpoint']['publication_status'] == 'staged'
        catalog.record_publication('a', 'published')
        page = catalog.event_page('run_a', after_sequence=snapshot['cursor']['after_sequence'])
        assert page['log_id'] == snapshot['cursor']['log_id']
        assert len(page['events']) == 1
        assert page['events'][0]['fields']['publication_status'] == 'published'
