import ast
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('native_vim',
    Path(__file__).parents[3]/'docs/e2e/tblite_native_vim_grader.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def source():
    return '\n'.join(['import subprocess, tempfile', 'def test_native():'] +
        ['    subprocess.run(cmd, capture_output=True, text=True, cwd="/app")'] * 11 +
        ['    tempfile.TemporaryDirectory(dir="/app")'] * 6 + ['    assert result == expected'])


def test_preserves_assertions_and_rewrites_all_boundaries():
    result = native.adapt(source())
    assert result.count('isolated_vim(cmd') == 11
    assert result.count("AgentTemporaryDirectory(dir='/app')") == 6
    assertions = lambda s: [ast.dump(n) for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Assert)]
    assert assertions(source()) == assertions(result)


def test_source_drift_fails_closed():
    with pytest.raises(ValueError, match='review required'):
        native.adapt(source() + '\nsubprocess.run(cmd)')


@pytest.mark.parametrize('args', [['sh', '-c', 'id'], ['vim'],
    ['vim', '-Es', '-u', 'NONE', '-n', '-S', '/tests/hidden.vim']])
def test_unreviewed_commands_refused(args):
    with pytest.raises(ValueError, match='Unreviewed'):
        native.isolated_vim(args, capture_output=True, text=True, cwd='/app')


def test_safe_path_refuses_link_writes(tmp_path):
    # Resolve platform aliases such as macOS /var -> /private/var first.
    tmp_path = tmp_path.resolve()
    target = tmp_path/'target'
    target.write_text('unchanged')
    link = tmp_path/'link'
    link.symlink_to(target)
    with pytest.raises(OSError):
        native.SafePath(link).write_text('overwrite')
    assert target.read_text() == 'unchanged'
    parent = tmp_path/'parent'
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        native.SafePath(parent/'target').read_text()


def test_safe_path_regular_file(tmp_path):
    path = native.SafePath(tmp_path.resolve()/'regular')
    path.write_text('hello')
    assert path.read_text() == 'hello'
