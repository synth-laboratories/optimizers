import io
import json
import zipfile

import pytest

from synth_optimizers.rl.daytona_workspace_control import scan, seal, restore, validate


def test_changed_deleted_roundtrip(tmp_path):
    agent, verifier = tmp_path/'agent', tmp_path/'verifier'
    for root in (agent, verifier):
        root.mkdir()
        (root/'same').write_text('same')
        (root/'delete').write_text('old')
        (root/'change').write_text('before')
    baseline = tmp_path/'baseline.json'
    baseline.write_text(json.dumps(scan(agent)))
    (agent/'delete').unlink()
    (agent/'change').write_text('after')
    (agent/'new').write_bytes(b'\0\xff')
    archive = tmp_path/'workspace.zip'
    expected = seal(agent, baseline, archive)
    assert restore(verifier, archive) == expected
    assert scan(agent) == scan(verifier)
    with zipfile.ZipFile(archive) as bundle:
        assert 'files/same' not in bundle.namelist()


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'fifo'])
def test_reject_unsafe_files(tmp_path, kind):
    import os
    root = tmp_path/'workspace'
    root.mkdir()
    outside = tmp_path/'outside'
    outside.write_text('private')
    if kind == 'symlink':
        (root/'bad').symlink_to(outside)
    elif kind == 'hardlink':
        os.link(outside, root/'bad')
    else:
        os.mkfifo(root/'bad')
    with pytest.raises(ValueError):
        scan(root)


def test_reject_extra_member_and_baseline_mismatch(tmp_path):
    root = tmp_path/'workspace'
    root.mkdir()
    baseline = tmp_path/'baseline.json'
    baseline.write_text('{}')
    archive = tmp_path/'workspace.zip'
    seal(root, baseline, archive)
    (root/'unexpected').write_text('x')
    with pytest.raises(ValueError, match='baseline'):
        restore(root, archive)
    with zipfile.ZipFile(archive, 'a') as bundle:
        bundle.writestr('files/../../escape', 'bad')
    with zipfile.ZipFile(archive) as bundle, pytest.raises(ValueError, match='unexpected'):
        validate(bundle)


def test_epoch_mtime_roundtrip_and_archive_determinism(tmp_path):
    import os
    root=tmp_path/'agent';root.mkdir()
    target=root/'artifact';target.write_bytes(b'reproducible bytes')
    baseline=tmp_path/'baseline.json';baseline.write_text('{}')
    first=tmp_path/'first.zip';second=tmp_path/'second.zip'
    os.utime(target,(0,0))
    seal(root,baseline,first)
    os.utime(target,(2_000_000_000,2_000_000_000))
    seal(root,baseline,second)
    assert first.read_bytes()==second.read_bytes()
    verifier=tmp_path/'verifier';verifier.mkdir()
    restore(verifier,first)
    assert (verifier/'artifact').read_bytes()==b'reproducible bytes'
