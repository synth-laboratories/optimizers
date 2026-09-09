"""Trusted, standard-library-only workspace bridge copied into task snapshots.

Run with isolated Python (-I). The agent UID cannot write this file or /root.
No archive extraction API is used: each member is validated and written explicitly.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 10000


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('unsafe workspace path')
    return p


def scan(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('workspace must be a real directory')
    rows, size = {}, 0
    for path in sorted(root.rglob('*')):
        mode = path.lstat()
        if stat.S_ISDIR(mode.st_mode):
            continue
        if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
            raise ValueError('only unlinked regular workspace files are transferable')
        size += mode.st_size
        if size > MAX_BYTES or len(rows) >= MAX_FILES:
            raise ValueError('workspace transfer limit exceeded')
        name = path.relative_to(root).as_posix()
        safe_name(name)
        rows[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def digest(rows):
    result = hashlib.sha256()
    for name, value in sorted(rows.items()):
        result.update(name.encode() + b'\0' + bytes.fromhex(value))
    return 'sha256:' + result.hexdigest()


def seal(root, baseline, archive):
    before = json.loads(Path(baseline).read_text())
    after = scan(root)
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    manifest = {'schema_version': 'rl.workspace_delta.v1', 'baseline': before,
                'final': after, 'changed': changed, 'deleted': sorted(set(before)-set(after)),
                'content_digest': digest(after)}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(zipfile.ZipInfo('manifest.json', (1980, 1, 1, 0, 0, 0)),
                        json.dumps(manifest, sort_keys=True), compress_type=zipfile.ZIP_DEFLATED)
        for name in changed:
            # The bridge transfers content, not source mtimes. Reproducible
            # builds often use Unix epoch timestamps, which ZIP cannot encode.
            bundle.writestr(zipfile.ZipInfo('files/'+name, (1980, 1, 1, 0, 0, 0)),
                            (Path(root)/name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return manifest


def validate(bundle):
    entries = bundle.infolist()
    if len(entries) > MAX_FILES+1 or sum(e.file_size for e in entries) > MAX_BYTES*2:
        raise ValueError('archive transfer limit exceeded')
    names = [e.filename for e in entries]
    if len(set(names)) != len(names):
        raise ValueError('duplicate archive members')
    manifest = json.loads(bundle.read('manifest.json'))
    for field in ('baseline', 'final', 'changed'):
        for name, value in manifest[field].items():
            safe_name(name)
            if len(value) != 64 or bytes.fromhex(value).hex() != value:
                raise ValueError('invalid content hash')
    for name in manifest['deleted']:
        safe_name(name)
    before, after = manifest['baseline'], manifest['final']
    if manifest['changed'] != {k:v for k,v in after.items() if before.get(k) != v}:
        raise ValueError('incorrect changed set')
    if manifest['deleted'] != sorted(set(before)-set(after)):
        raise ValueError('incorrect deleted set')
    if set(names) != {'manifest.json'} | {'files/'+k for k in manifest['changed']}:
        raise ValueError('unexpected archive members')
    for name, expected in manifest['changed'].items():
        if hashlib.sha256(bundle.read('files/'+name)).hexdigest() != expected:
            raise ValueError('transfer hash mismatch')
    if digest(after) != manifest['content_digest']:
        raise ValueError('final manifest hash mismatch')
    return manifest


def restore(root, archive):
    root = Path(root)
    with zipfile.ZipFile(archive) as bundle:
        manifest = validate(bundle)
        if scan(root) != manifest['baseline']:
            raise ValueError('verifier baseline differs from agent baseline')
        for name in manifest['deleted']:
            (root/name).unlink()
        for name in manifest['changed']:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read('files/'+name))
        if scan(root) != manifest['final']:
            raise ValueError('verifier reconstructed workspace differs')
    return manifest


if __name__ == '__main__':
    operation, root, artifact = sys.argv[1:4]
    if operation == 'baseline':
        Path(artifact).write_text(json.dumps(scan(root), sort_keys=True))
    elif operation == 'seal':
        print(json.dumps(seal(root, artifact, sys.argv[4]), sort_keys=True))
    elif operation == 'restore':
        print(json.dumps(restore(root, artifact), sort_keys=True))
    else:
        raise ValueError('unknown workspace operation')
