"""Provider-free read models shared by CLI and future service consumers.

These views deliberately do not attest provider availability or authorize resume.
Actual resume continues through the existing resolver and compatibility gates.
"""
from __future__ import annotations

from dataclasses import asdict

from .catalog import CheckpointCatalog


def capabilities() -> dict:
    return {
        'schema_version': 'rl_read_capabilities.v1',
        'checkpoint_details': True,
        'checkpoint_events': True,
        'checkpoint_snapshot': True,
        'event_delivery': 'at_least_once_cursor_polling',
        'event_scope': 'checkpoint_registration_publication_save_alias_availability',
        'historical_event_backfill': False,
        'provider_availability_checks': True,
        'provider_check_mode': 'explicit_metadata_verification',
        'retention': 'keep_all_no_automatic_pruning',
        'experiment_orchestration': False,
        'remote_controls': False,
    }


def run_snapshot(catalog: CheckpointCatalog, run_id: str) -> dict:
    """Consistent current views + cursor; subscribe strictly after that cursor.

    The catalog transaction briefly fences writers while materializing the
    snapshot. No provider calls or network waits belong inside this boundary.
    """
    with catalog.transaction():
        cursor = catalog.event_head(run_id)
        checkpoints = [checkpoint_details(catalog, view.checkpoint_id)
                       for view in catalog.list_checkpoints(run_id=run_id)]
        return {'schema_version': 'rl_checkpoint_snapshot.v1', 'run_id': run_id,
                'cursor': cursor, 'checkpoints': checkpoints}


def checkpoint_details(catalog: CheckpointCatalog, checkpoint_id: str) -> dict:
    view = catalog.describe_checkpoint(checkpoint_id)
    record = view.record
    observations = catalog.artifact_observations(checkpoint_id)
    return {
        'schema_version': 'rl_checkpoint_details.v1',
        'checkpoint': view.to_payload(),
        'ancestry': list(catalog.ancestry(checkpoint_id)),
        'publication_history': [
            {'status': status, 'recorded_at': timestamp, 'reason': reason}
            for status, timestamp, reason in catalog.publication_history(checkpoint_id)
        ],
        'aliases': [asdict(alias) for alias in catalog.list_aliases()
                    if alias.target_kind == 'checkpoint' and alias.target_id == checkpoint_id],
        'alias_history': list(catalog.alias_history(checkpoint_id)),
        'artifact_health': (observations[-1] if observations else {'status': 'unverified', 'checked_at': None}),
        'resume': {
            'has_training_state': record.is_resumable,
            'eligible': False,
            'reason': ('provider_and_compatibility_verification_required'
                       if record.is_resumable else 'training_state_missing'),
        },
    }


def verify_checkpoint(catalog, checkpoint_id, provider):
    """Persist provider metadata observations; compatibility is still checked on resume."""
    record = catalog.describe_checkpoint(checkpoint_id).record
    roles = {}
    for role in ('sampler_weights', 'training_state'):
        artifact = getattr(record.artifacts, role, None)
        if artifact is not None:
            roles[role] = provider.describe_artifact(artifact.ref)
    catalog.record_artifact_observation(checkpoint_id, {'artifacts': roles,
        'verification': 'provider_metadata_not_downloaded_content'})
    return checkpoint_details(catalog, checkpoint_id)
