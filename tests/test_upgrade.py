"""Fake-process upgrade tests: no Docker daemon or deployment is invoked."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/upgrade.py'
SPEC = importlib.util.spec_from_file_location('vision_upgrade_tests_target', MODULE_PATH)
upgrade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upgrade)
OLD_IMAGE = 'image2ppt-vision:0.1.0'
NEW_IMAGE = 'image2ppt-vision:0.3.0'
OLD_ID = 'sha256:' + '1' * 64
NEW_ID = 'sha256:' + '2' * 64
TOKEN = 'private-fixture-token-do-not-print'


class FakeDocker:
    def __init__(self, previous, release):
        self.previous, self.release = previous, release
        self.evidence = None
        self.calls = []
        self.fail_tests = False
        self.fail_cutover = False
        self.fail_health = False
        self.wrong_live_version = False
        self.change_other = False
        self.extra_compose_service = False
        self.previous_compose_images = [OLD_IMAGE]
        self.new_compose_images = [NEW_IMAGE]
        self.ids = {OLD_IMAGE: OLD_ID, NEW_IMAGE: NEW_ID}
        self.state = {}
        for i, service in enumerate(upgrade.SERVICES):
            identifier = str(i + 3) * 64
            self.state[identifier] = {'id': identifier,
                'name': upgrade.PROJECT + '-' + service + '-1',
                'image_id': OLD_ID, 'image': OLD_IMAGE, 'running': True,
                'project': upgrade.PROJECT, 'service': service,
                'config_files': str(previous / 'compose.yaml')}
        # The nine existing GPU services are ordinary non-target records here.
        # Both stopped and running states must be preserved, with identical IDs.
        for i in range(9):
            identifier = f'{i+10:064x}'
            self.state[identifier] = {'id': identifier, 'name': f'existing-gpu-{i}',
                'image_id': 'sha256:' + '9'*64, 'image': 'unrelated:existing',
                'running': i % 2 == 0, 'project': 'neighbor-project',
                'service': f'neighbor-{i}', 'config_files': '/unrelated/compose.yaml'}
        self.initial = deepcopy(self.state)

    def factory(self, evidence):
        self.evidence = Path(evidence)
        return self

    def version(self):
        current = next(x for x in self.state.values()
                       if x['project'] == upgrade.PROJECT and x['service'] == 'api')
        return '0.3.0' if current['image_id'] == NEW_ID else '0.1.0'

    def command(self, arguments, *, label, env=None, timeout=120):
        arguments = list(arguments)
        self.calls.append({'args': arguments, 'label': label, 'env': env, 'timeout': timeout})
        upgrade.private_write(self.evidence / f'fake-{len(self.calls):03d}.log', label + '\n')
        if arguments[:3] == ['docker', 'ps', '-aq']:
            return '\n'.join(self.state) + '\n'
        if arguments[:2] == ['docker', 'inspect']:
            assert arguments[2:4] == ['--format', upgrade.INSPECT_FORMAT]
            return '\n'.join(json.dumps(self.state[i]) for i in arguments[4:]) + '\n'
        if arguments[:3] == ['docker', 'image', 'inspect']:
            return self.ids[arguments[-1]] + '\n'
        if arguments[:2] == ['docker', 'exec']:
            assert arguments[3:] == ['python', '-c', upgrade.VERSION_CODE]
            item = self.state[arguments[2]]
            version = '0.3.0' if item['image_id'] == NEW_ID else '0.1.0'
            if self.wrong_live_version and version == '0.3.0':
                version = '0.1.9'
            return version + '\n'
        if arguments[:2] == ['docker', 'run']:
            assert '--rm' in arguments and '--network=none' in arguments and '--pull=never' in arguments
            assert '--gpus' not in arguments and '-v' not in arguments
            assert NEW_ID in arguments
            if '-m' in arguments:
                assert arguments[arguments.index('-m'):] == [
                    '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider']
                if self.fail_tests:
                    raise upgrade.UpgradeError('candidate-offline-tests: command exited 1')
                return 'synthetic candidate tests passed\n'
            return '0.3.0\n'
        assert arguments[:2] == ['docker', 'compose'], arguments
        assert arguments[arguments.index('--project-name')+1] == upgrade.PROJECT
        files = [Path(arguments[i+1]) for i, a in enumerate(arguments) if a == '--file']
        is_new = files[0] == self.release / 'compose.yaml'
        if 'config' in arguments:
            if '--services' in arguments:
                return 'api\ningress\n' + ('unexpected\n' if self.extra_compose_service and is_new else '')
            if '--images' in arguments:
                return '\n'.join(self.new_compose_images if is_new else self.previous_compose_images) + '\n'
            assert '--quiet' in arguments
            return ''
        assert 'up' in arguments
        assert arguments[-2:] == ['api', 'ingress']
        assert '--no-build' in arguments and '--no-deps' in arguments
        assert arguments[arguments.index('--pull')+1] == 'never'
        assert not any(a in arguments for a in ('stop', 'down', 'restart', 'rm'))
        if not is_new:
            assert len(files) == 2
            pinned = json.loads(files[1].read_text(encoding='utf-8'))
            assert pinned == {'services': {'api': {'image': OLD_ID}, 'ingress': {'image': OLD_ID}}}
        current = upgrade.target_records(self.state)
        for i, service in enumerate(upgrade.SERVICES):
            self.state.pop(current[service]['id'])
            item = deepcopy(current[service])
            identifier = str(i + (5 if is_new else 7)) * 64
            item.update(id=identifier, image_id=NEW_ID if is_new else OLD_ID,
                        image=NEW_IMAGE if is_new else OLD_ID, running=True,
                        config_files=','.join(map(str, files)))
            self.state[identifier] = item
        if is_new and self.change_other:
            neighbor = next(x for x in self.state.values() if x['project'] != upgrade.PROJECT)
            neighbor['running'] = not neighbor['running']
        if is_new and self.fail_cutover:
            raise upgrade.UpgradeError('upgrade-targets: command exited 1')
        return ''


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / 'deployment'
    previous = root / 'releases' / 'old'
    release = root / 'releases' / 'new'
    for directory, image in ((previous, OLD_IMAGE), (release, NEW_IMAGE)):
        directory.mkdir(parents=True)
        (directory / 'compose.yaml').write_text(
            f'services:\n  api:\n    image: {image}\n  ingress:\n    image: {image}\n', encoding='utf-8')
    (previous / 'runtime.env').write_text('VISION_TOKEN=' + TOKEN + '\n', encoding='utf-8')
    fake = FakeDocker(previous, release)
    clock = {'now': 0.0}
    def monotonic():
        clock['now'] += .1
        return clock['now']
    monkeypatch.setattr(upgrade.time, 'monotonic', monotonic)
    monkeypatch.setattr(upgrade.time, 'sleep', lambda duration: clock.update(now=clock['now']+duration))
    monkeypatch.setattr(upgrade, 'probe_health',
                        lambda timeout: not fake.fail_health or fake.version() == '0.1.0')
    return SimpleNamespace(root=root, previous=previous, release=release, fake=fake)


def execute(case, **kwargs):
    return upgrade.upgrade(case.release, case.previous, NEW_IMAGE,
                           root=case.root, runner_factory=case.fake.factory, **kwargs)


def result_record(case):
    return json.loads((case.fake.evidence / 'upgrade.json').read_text(encoding='utf-8'))


def up_calls(case):
    return [c for c in case.fake.calls if c['args'][:2] == ['docker', 'compose'] and 'up' in c['args']]


def assert_no_other_mutation_commands(case):
    for call in case.fake.calls:
        args = call['args']
        assert not any(part in args for part in ('stop', 'down', 'restart', 'rm', '--gpus'))
        assert 'gpu_services.py' not in ' '.join(args)
        if 'up' in args:
            assert args[-2:] == ['api', 'ingress']
            assert args[args.index('--project-name')+1] == upgrade.PROJECT


def test_upgrade_preflights_offline_and_preserves_all_nine_other_states(case, capsys):
    result = execute(case)
    assert result['status'] == 'PASS' and result['observed_version'] == '0.3.0'
    assert result['previous_version'] == '0.1.0'
    assert result['non_target_unchanged'] and result['non_target_containers'] == 9
    assert upgrade.other_states(case.fake.state) == upgrade.other_states(case.fake.initial)
    assert (case.release / 'runtime.env').read_bytes() == (case.previous / 'runtime.env').read_bytes()
    assert result['runtime'] == {'copied_from_previous': True, 'token_preserved': True}
    labels = [c['label'] for c in case.fake.calls]
    assert labels.index('candidate-offline-tests') < labels.index('upgrade-targets')
    assert labels.index('candidate-offline-version') < labels.index('upgrade-targets')
    assert len(up_calls(case)) == 1
    assert_no_other_mutation_commands(case)
    assert capsys.readouterr().out == ''
    for path in case.fake.evidence.iterdir():
        assert TOKEN not in path.read_text(encoding='utf-8')
        if os.name == 'posix':
            assert path.stat().st_mode & 0o777 == 0o600
    if os.name == 'posix':
        assert (case.release / 'runtime.env').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('failure', ['fail_tests', 'extra_compose_service'])
def test_preflight_failure_never_restarts_any_service(case, failure):
    setattr(case.fake, failure, True)
    with pytest.raises(upgrade.UpgradeError) as raised:
        execute(case)
    result = result_record(case)
    assert raised.value.evidence == str(case.fake.evidence)
    assert result['status'] == 'FAILED_BEFORE_CUTOVER'
    assert result['rollback'] == 'not_required'
    assert not up_calls(case)
    assert case.fake.state == case.fake.initial
    assert_no_other_mutation_commands(case)


@pytest.mark.parametrize('failure', ['fail_cutover', 'fail_health', 'wrong_live_version'])
def test_cutover_failure_rolls_back_only_targets_to_recorded_images(case, failure):
    setattr(case.fake, failure, True)
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    result = result_record(case)
    assert result['status'] == 'FAILED_AFTER_CUTOVER'
    assert result['rollback'] == 'restored_previous_target_images'
    assert result['rollback_version'] == '0.1.0'
    assert result['non_target_unchanged']
    assert upgrade.other_states(case.fake.state) == upgrade.other_states(case.fake.initial)
    calls = up_calls(case)
    assert len(calls) == 2
    assert str(case.previous / 'compose.yaml') in calls[1]['args']
    assert str(case.release / 'compose.yaml') not in calls[1]['args']
    assert '--no-build' in calls[1]['args']
    assert {t['image_id'] for t in upgrade.target_records(case.fake.state).values()} == {OLD_ID}
    assert_no_other_mutation_commands(case)


def test_detected_external_state_change_is_recorded_without_mutating_that_container(case):
    case.fake.change_other = True
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    result = result_record(case)
    assert not result['non_target_unchanged']
    assert result['rollback'] == 'restored_previous_target_images'
    assert 'non-target container' in result['failure']
    assert len(up_calls(case)) == 2
    assert_no_other_mutation_commands(case)


def test_previous_release_must_own_the_running_target_containers(case):
    api = upgrade.target_records(case.fake.state)['api']
    api['config_files'] = str(case.root / 'releases' / 'different' / 'compose.yaml')
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    assert 'does not own' in result_record(case)['failure']
    assert not up_calls(case)
    assert not (case.release / 'runtime.env').exists()


def test_previous_compose_must_resolve_to_running_image_identity(case):
    case.fake.ids[OLD_IMAGE] = 'sha256:' + 'f'*64
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    assert 'do not match current target images' in result_record(case)['failure']
    assert not up_calls(case)


def test_new_compose_must_use_explicit_candidate_image(case):
    case.fake.new_compose_images = ['unexpected:tag']
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    assert 'explicit candidate' in result_record(case)['failure']
    assert not up_calls(case)


def test_existing_new_runtime_is_preserved_when_token_matches(case):
    contents = ('VISION_TOKEN=' + TOKEN + '\nVISION_FONT_MANIFEST=/fonts/manifest.json\n').encode()
    (case.release / 'runtime.env').write_bytes(contents)
    result = execute(case)
    assert not result['runtime']['copied_from_previous']
    assert (case.release / 'runtime.env').read_bytes() == contents


def test_changed_token_is_rejected_without_overwriting_existing_file(case):
    contents = b'VISION_TOKEN=different-fixture-token\n'
    (case.release / 'runtime.env').write_bytes(contents)
    with pytest.raises(upgrade.UpgradeError):
        execute(case)
    assert 'changes the existing service token' in result_record(case)['failure']
    assert (case.release / 'runtime.env').read_bytes() == contents
    assert not up_calls(case)


def test_explicit_font_manifest_reaches_only_new_compose_environment(case):
    result = execute(case, font_manifest='/fonts/manifest.json')
    assert result['status'] == 'PASS'
    for call in case.fake.calls:
        if call['label'].startswith('candidate-') and call['args'][:2] == ['docker', 'compose'] \
                or call['label'] == 'upgrade-targets':
            assert call['env']['VISION_FONT_MANIFEST'] == '/fonts/manifest.json'
        else:
            assert call['env'] is None


@pytest.mark.parametrize('which', ['release', 'previous'])
@pytest.mark.parametrize('form', ['parent-traversal', 'outside', 'releases-root', 'relative'])
def test_release_boundaries_are_rejected_before_docker(case, which, form):
    if form == 'parent-traversal':
        path = case.root / 'releases' / '..' / 'releases' / 'new'
    elif form == 'outside':
        path = case.root / 'outside'
        path.mkdir()
    elif form == 'releases-root':
        path = case.root / 'releases'
    else:
        path = Path('releases/new')
    release = path if which == 'release' else case.release
    previous = path if which == 'previous' else case.previous
    with pytest.raises(upgrade.UpgradeError):
        upgrade.upgrade(release, previous, NEW_IMAGE, root=case.root,
                        runner_factory=case.fake.factory)
    assert not case.fake.calls


@pytest.mark.parametrize('manifest', ['/etc/passwd', '/fonts/../secrets', 'fonts/manifest.json'])
def test_font_manifest_path_is_bounded_before_docker(case, manifest):
    with pytest.raises(upgrade.UpgradeError):
        execute(case, font_manifest=manifest)
    assert not case.fake.calls


def test_same_release_is_rejected_before_docker(case):
    with pytest.raises(upgrade.UpgradeError, match='distinct'):
        upgrade.upgrade(case.release, case.release, NEW_IMAGE,
                        root=case.root, runner_factory=case.fake.factory)
    assert not case.fake.calls


def test_runner_uses_argument_arrays_private_logs_and_never_dumps_environment(tmp_path, monkeypatch, capsys):
    def fake_run(arguments, **kwargs):
        assert arguments == ['docker', 'version']
        assert kwargs['shell'] is False and kwargs['check'] is False
        assert kwargs['stdout'] == upgrade.subprocess.PIPE
        assert kwargs['stderr'] == upgrade.subprocess.STDOUT
        assert kwargs['env'] == {'PRIVATE_SENTINEL': TOKEN}
        return SimpleNamespace(returncode=0, stdout='safe command output\n')
    monkeypatch.setattr(upgrade.subprocess, 'run', fake_run)
    runner = upgrade.Runner(tmp_path)
    assert runner.command(['docker', 'version'], label='version',
                          env={'PRIVATE_SENTINEL': TOKEN}) == 'safe command output\n'
    logs = list(tmp_path.glob('*.log'))
    assert len(logs) == 1 and logs[0].read_text() == 'safe command output\n'
    assert TOKEN not in logs[0].read_text()
    assert capsys.readouterr().out == ''


def test_cli_is_explicit_and_defaults_to_version_021(monkeypatch, capsys):
    calls = []
    def fake_upgrade(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'PASS', 'observed_version': '0.3.0',
                'non_target_unchanged': True, 'evidence': '/private/evidence'}
    monkeypatch.setattr(upgrade, 'upgrade', fake_upgrade)
    assert upgrade.main(['--release', '/data/image2ppt-vision/releases/new',
        '--previous-release', '/data/image2ppt-vision/releases/old', '--image', NEW_IMAGE]) == 0
    assert calls[0][1]['expected_version'] == '0.3.0'
    assert json.loads(capsys.readouterr().out)['non_target_unchanged'] is True
