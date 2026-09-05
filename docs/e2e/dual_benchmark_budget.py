"""Cross-process, fail-closed token-charge reservations for the dual pilot."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path('/Users/joshuapurtell/GitHub/optimizers/temp/healthbench_craftax_uplift_20260904')


def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / 'budget.sqlite3', timeout=60)
    db.execute('CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY, lane TEXT, reserved REAL, counted REAL, usage TEXT, started REAL, finished REAL)')
    return db


def reserve(lane, upper):
    if upper < 0:
        raise ValueError('negative reservation')
    # Leave room to flush local evidence if concurrent work fills this disk.
    # Refuse before the paid call, not after its checkpoint publication fails.
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError('paid call refused: less than 2 GiB free for durable evidence')
    key = uuid.uuid4().hex
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        used = db.execute('SELECT COALESCE(SUM(COALESCE(counted,reserved)),0) FROM charges').fetchone()[0]
        # One dollar of the $49 aggregate is held aside for non-token overhead.
        if used + upper > 48:
            raise RuntimeError(f'aggregate paid budget exhausted: {used:.4f} + {upper:.4f} > 48 token dollars')
        db.execute('INSERT INTO charges VALUES (?,?,?,?,?,?,?)', (key,lane,upper,None,None,time.time(),None))
    return key


def settle(key, counted, usage):
    with connect() as db:
        reserved = db.execute('SELECT reserved FROM charges WHERE id=?',(key,)).fetchone()[0]
        if counted > reserved + 1e-8:
            raise RuntimeError(f'provider usage exceeded reservation: {counted} > {reserved}')
        db.execute('UPDATE charges SET counted=?,usage=?,finished=? WHERE id=?',(counted,json.dumps(usage),time.time(),key))


def load_credentials(*names):
    # Both project-local sources are explicitly authorized. No Keychain.
    values = {}
    for name in names:
        path = Path('/Users/joshuapurtell/GitHub/evals/.env' if name == 'OPENROUTER_API_KEY' else '/Users/joshuapurtell/GitHub/frontend/.env.local')
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and key.strip() == name:
                values[name] = value.strip().strip('\"\'')
        if name == 'OPENROUTER_API_KEY' or not os.environ.get(name):
            os.environ[name] = values.get(name, '')
        if not os.environ[name]:
            raise RuntimeError(f'authorized credential unavailable: {name}')


def guard_provider(provider, observed=None, observed_path=None):
    provider.max_attempts = 1  # Ambiguous failures retain the full reservation.
    if observed is not None:
        save = provider.save_checkpoint
        def save_observed(*args, **kwargs):
            result = save(*args, **kwargs)
            observed[result.provider_reference] = result.digest
            if observed_path:
                from screen_banking77 import _write_json
                _write_json(observed_path, observed)
            return result
        provider.save_checkpoint = save_observed
    for method in ('sample', 'sample_checkpoint', 'train_step', 'forward'):
        original = getattr(provider, method)

        def guarded(identity, request, _method=method, _original=original):
            if _method in ('sample', 'sample_checkpoint'):
                upper = (len(request.prompt_token_ids)*.18 + request.max_tokens*.45)/1e6
            elif _method == 'train_step':
                upper = sum(len(row.get('token_ids') or row.get('input_ids') or ()) + len(row.get('prompt_token_ids') or ()) for row in request.data)*.396/1e6
            else:
                upper = sum(len(row) for row in request.token_ids)*.396/1e6
            key = reserve('tinker:'+_method, upper)
            result = _original(identity, request)
            usage = result.usage
            if _method in ('sample', 'sample_checkpoint'):
                counted = (len(request.prompt_token_ids)*.18 + len(result.token_ids)*.45)/1e6
            else:
                counted = upper
            settle(key, counted, {'input_tokens':usage.input_tokens,'output_tokens':usage.output_tokens,'training_tokens':usage.training_tokens,'provider_cost':usage.cost_usd})
            return result

        setattr(provider, method, guarded)
    return provider


def paid(config=None, **kwargs):
    import paid_plane
    from synth_optimizers.rl.resolver import MappingArtifactProbe
    path = Path(config.artifacts.catalog).parent / 'artifact_digests.json'
    observed = json.loads(path.read_text()) if path.exists() else {}
    kwargs.setdefault('artifact_probe', MappingArtifactProbe(digests=observed))
    original = paid_plane.build_provider
    paid_plane.build_provider = lambda cfg: guard_provider(original(cfg), observed, path)
    try:
        return paid_plane.paid(config=config, **kwargs)
    finally:
        paid_plane.build_provider = original
