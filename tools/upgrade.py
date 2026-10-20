"""Explicit, reversible upgrade of the existing image2ppt-vision-v1 project.

Never stops GPU owners or changes another Compose project. The candidate image
must already exist locally. Runtime credentials are copied without disclosure;
all command output and inventory evidence are private files on the host.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time
import urllib.error
import urllib.request
from uuid import uuid4


ROOT = Path('/data/image2ppt-vision')
PROJECT = 'image2ppt-vision-v1'
SERVICES = ('api', 'ingress')
HEALTH_URL = 'http://127.0.0.1:18761/health'
VERSION_CODE = 'import image2ppt_vision; print(image2ppt_vision.__version__)'
INSPECT_FORMAT = ('{"id":{{json .Id}},"name":{{json .Name}},'
    '"image_id":{{json .Image}},"image":{{json .Config.Image}},'
    '"running":{{json .State.Running}},'
    '"project":{{json (index .Config.Labels "com.docker.compose.project")}},'
    '"service":{{json (index .Config.Labels "com.docker.compose.service")}},'
    '"config_files":{{json (index .Config.Labels "com.docker.compose.project.config_files")}}}')


class UpgradeError(RuntimeError):
    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = str(evidence) if evidence else None


def private_write(path, content):
    """Create one private evidence file without overwriting an older record."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        if os.name == 'posix':
            os.fchmod(handle.fileno(), 0o600)
        handle.write(content if isinstance(content, bytes) else content.encode('utf-8'))


def record(path, value):
    private_write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Runner:
    """Argument arrays only; raw stdout never reaches the interactive terminal."""
    def __init__(self, evidence):
        self.evidence = Path(evidence)
        self.number = 0

    def command(self, arguments, *, label, env=None, timeout=120):
        self.number += 1
        logfile = self.evidence / f'{self.number:03d}-{label}.log'
        try:
            result = subprocess.run(list(arguments), shell=False, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or b''
            private_write(logfile, output)
            raise UpgradeError(f'{label}: command timed out') from None
        private_write(logfile, result.stdout or '')
        if result.returncode:
            raise UpgradeError(f'{label}: command exited {result.returncode}')
        return result.stdout or ''


def release_path(value, root):
    raw = Path(value)
    if not raw.is_absolute() or '..' in raw.parts:
        raise UpgradeError('release must be an absolute path without traversal')
    releases = (root / 'releases').resolve(strict=True)
    resolved = raw.resolve(strict=True)
    if resolved == releases or not resolved.is_relative_to(releases) or not resolved.is_dir():
        raise UpgradeError('release is outside the deployment releases directory')
    compose = resolved / 'compose.yaml'
    if compose.is_symlink() or not compose.is_file() or not compose.resolve().is_relative_to(resolved):
        raise UpgradeError('release requires a regular local compose.yaml')
    return resolved


def read_runtime(path):
    if path.is_symlink() or not path.is_file():
        raise UpgradeError('runtime.env must be an existing regular file')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise UpgradeError('runtime.env must be a regular file')
        data = handle.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise UpgradeError('runtime.env exceeds the supported size')
    return data


def runtime_token(content):
    try:
        lines = content.decode('utf-8-sig').splitlines()
    except UnicodeError:
        raise UpgradeError('runtime.env must be UTF-8') from None
    values = []
    for line in lines:
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        if separator and key.strip() == 'VISION_TOKEN':
            values.append(value.strip().strip('"\''))
    if len(values) != 1 or not values[0]:
        raise UpgradeError('runtime.env must contain one nonempty VISION_TOKEN')
    return values[0]


def preserve_runtime(previous, release):
    old_bytes = read_runtime(previous / 'runtime.env')
    old_token = runtime_token(old_bytes)
    new_path = release / 'runtime.env'
    if new_path.is_symlink():
        raise UpgradeError('new runtime.env cannot be a symbolic link')
    copied = not new_path.exists()
    if copied:
        private_write(new_path, old_bytes)
    new_bytes = read_runtime(new_path)
    if runtime_token(new_bytes) != old_token:
        raise UpgradeError('new runtime.env changes the existing service token')
    if os.name == 'posix':
        os.chmod(new_path, 0o600, follow_symlinks=False)
    return {'copied_from_previous': copied, 'token_preserved': True}


def inventory(runner):
    identifiers = runner.command(['docker', 'ps', '-aq'], label='container-ids').split()
    if not identifiers:
        return {}
    output = runner.command(['docker', 'inspect', '--format', INSPECT_FORMAT, *identifiers],
                            label='container-inventory')
    records = [json.loads(line) for line in output.splitlines() if line.strip()]
    result = {}
    for item in records:
        item['name'] = item['name'].lstrip('/')
        if not isinstance(item['running'], bool) or item['id'] in result:
            raise UpgradeError('invalid container inventory')
        result[item['id']] = item
    if set(result) != set(identifiers):
        # docker ps may return short IDs; require inspect to match each prefix.
        if len(result) != len(identifiers) or any(
                not any(full.startswith(short) for full in result) for short in identifiers):
            raise UpgradeError('container inventory changed during inspection')
    return result


def target_records(snapshot, *, require_running=False):
    targets = [item for item in snapshot.values() if item['project'] == PROJECT]
    if len(targets) != 2 or {item['service'] for item in targets} != set(SERVICES):
        raise UpgradeError('the existing project must contain exactly api and ingress')
    if require_running and not all(item['running'] for item in targets):
        raise UpgradeError('existing api and ingress must be running before upgrade')
    return {item['service']: item for item in targets}


def other_states(snapshot):
    return {identifier: {'name': item['name'], 'running': item['running']}
            for identifier, item in snapshot.items() if item['project'] != PROJECT}


def unchanged_others(before, after):
    return other_states(before) == other_states(after)


def compose(release, override=None):
    command = ['docker', 'compose', '--project-name', PROJECT,
               '--file', str(release / 'compose.yaml')]
    if override is not None:
        command += ['--file', str(override)]
    return command


def compose_images(runner, command, label, *, env=None):
    runner.command([*command, 'config', '--quiet'], label=f'{label}-config', env=env)
    services = runner.command([*command, 'config', '--services'],
                             label=f'{label}-services', env=env).split()
    if len(services) != 2 or set(services) != set(SERVICES):
        raise UpgradeError('compose may declare only api and ingress services')
    images = runner.command([*command, 'config', '--images'],
                           label=f'{label}-images', env=env).split()
    if not images:
        raise UpgradeError('compose must use prebuilt images')
    return set(images)


def image_id(runner, image, label):
    result = runner.command(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                            label=label).strip()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', result):
        raise UpgradeError('image inspection did not return one local image ID')
    return result


def package_version(runner, identifier, label, *, timeout=20):
    value = runner.command(['docker', 'exec', identifier, 'python', '-c', VERSION_CODE],
                           label=label, timeout=timeout).strip()
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-+.][A-Za-z0-9.-]+)?', value):
        raise UpgradeError('container did not return a valid package version')
    return value


def probe_health(timeout):
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
            return response.status == 200 and json.load(response).get('status') == 'alive'
    except (OSError, urllib.error.URLError, ValueError):
        return False


def wait_healthy(runner, identifier, expected, label, *, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if probe_health(min(2, remaining)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                actual = package_version(runner, identifier, f'{label}-version',
                                         timeout=min(10, remaining))
            except UpgradeError:
                actual = None
            if actual is not None:
                if actual != expected:
                    raise UpgradeError(f'{label}: package version mismatch')
                return actual
        time.sleep(min(.5, max(0, deadline-time.monotonic())))
    raise UpgradeError(f'{label}: health and version did not pass within {seconds} seconds')


def upgrade(release, previous_release, image, *, expected_version='0.2.1',
            font_manifest=None, root=None, runner_factory=Runner):
    """Run an explicitly requested upgrade. Root injection is for isolated tests."""
    root = Path(root) if root is not None else ROOT
    release = release_path(release, root)
    previous = release_path(previous_release, root)
    if release == previous:
        raise UpgradeError('new and previous releases must be distinct')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*', image):
        raise UpgradeError('invalid candidate image reference')
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-+.][A-Za-z0-9.-]+)?', expected_version):
        raise UpgradeError('invalid expected package version')
    new_environment = None
    if font_manifest is not None:
        candidate = PurePosixPath(font_manifest)
        if not candidate.is_absolute() or '..' in candidate.parts or not candidate.is_relative_to('/fonts'):
            raise UpgradeError('font manifest must be an absolute path beneath /fonts')
        new_environment = os.environ.copy()
        new_environment['VISION_FONT_MANIFEST'] = str(candidate)
    evidence_root = root / 'evidence'
    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence = evidence_root / ('upgrade-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                               + '-' + uuid4().hex[:8])
    evidence.mkdir(mode=0o700)
    runner = runner_factory(evidence)
    old_command, new_command = compose(previous), compose(release)
    old_targets = None
    before = None
    old_version = None
    rollback_command = None
    cutover_started = False
    outcome = {'status': 'FAILED_BEFORE_CUTOVER', 'project': PROJECT,
               'services': list(SERVICES), 'release': str(release),
               'previous_release': str(previous), 'candidate_image': image,
               'expected_version': expected_version, 'evidence': str(evidence)}
    try:
        before = inventory(runner)
        record(evidence / 'containers-before.json', before)
        old_targets = target_records(before, require_running=True)
        for item in old_targets.values():
            config_files = [Path(p).resolve() for p in (item['config_files'] or '').split(',') if p]
            if previous / 'compose.yaml' not in config_files:
                raise UpgradeError('previous release does not own the current target containers')
        record(evidence / 'target-before.json', old_targets)
        old_version = package_version(runner, old_targets['api']['id'], 'previous-version')
        outcome['previous_version'] = old_version
        previous_images = compose_images(runner, old_command, 'previous')
        previous_ids = {image_id(runner, value, 'previous-image') for value in previous_images}
        if any(item['image_id'] not in previous_ids for item in old_targets.values()):
            raise UpgradeError('previous compose images do not match current target images')
        candidate_id = image_id(runner, image, 'candidate-image')
        if candidate_id in previous_ids:
            raise UpgradeError('candidate image must differ from the running previous image')
        outcome['candidate_image_id'] = candidate_id
        outcome['runtime'] = preserve_runtime(previous, release)
        if compose_images(runner, new_command, 'candidate', env=new_environment) != {image}:
            raise UpgradeError('new compose images differ from the explicit candidate image')
        # Pin rollback to the image IDs actually running, independent of tag drift.
        override = evidence / 'rollback-images.json'
        record(override, {'services': {service: {'image': item['image_id']}
                                      for service, item in old_targets.items()}})
        rollback_command = compose(previous, override)
        runner.command([*rollback_command, 'config', '--quiet'], label='rollback-config')
        record(evidence / 'release-files.json', {
            'new_compose_sha256': hashlib.sha256((release/'compose.yaml').read_bytes()).hexdigest(),
            'previous_compose_sha256': hashlib.sha256((previous/'compose.yaml').read_bytes()).hexdigest(),
            'font_manifest': font_manifest})
        offline = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--cpus=8', '--memory=16g',
                   '--entrypoint=python', candidate_id]
        runner.command([*offline, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider'],
                       label='candidate-offline-tests', timeout=1800)
        version = runner.command([*offline, '-c', VERSION_CODE],
                                 label='candidate-offline-version', timeout=60).strip()
        if version != expected_version:
            raise UpgradeError('candidate image package version mismatch')
        checked = inventory(runner)
        record(evidence / 'containers-pre-cutover.json', checked)
        if checked != before:
            raise UpgradeError('container state changed during preflight')
        if image_id(runner, image, 'candidate-image-recheck') != candidate_id:
            raise UpgradeError('candidate image tag changed during preflight')
        # No stop/down, no GPU-owner actions, no implicit builds/pulls/dependencies.
        cutover_started = True
        runner.command([*new_command, 'up', '-d', '--no-build', '--pull', 'never',
                        '--no-deps', *SERVICES], label='upgrade-targets', env=new_environment)
        started = inventory(runner)
        new_targets = target_records(started)
        if any(item['image_id'] != candidate_id for item in new_targets.values()):
            raise UpgradeError('upgraded target image IDs do not match the tested candidate')
        actual = wait_healthy(runner, new_targets['api']['id'], expected_version, 'upgrade')
        after = inventory(runner)
        record(evidence / 'containers-after.json', after)
        if not unchanged_others(before, after):
            raise UpgradeError('non-target container identity or running state changed')
        outcome.update(status='PASS', observed_version=actual,
                       non_target_unchanged=True,
                       non_target_containers=len(other_states(before)), rollback='not_needed')
        record(evidence / 'upgrade.json', outcome)
        return outcome
    except BaseException as failure:
        outcome['failure_type'] = type(failure).__name__
        # All UpgradeError messages are fixed labels; raw subprocess text is private.
        outcome['failure'] = str(failure) if isinstance(failure, UpgradeError) else 'unexpected upgrade failure'
        outcome['rollback'] = 'not_required'
        if cutover_started and rollback_command is not None:
            outcome['status'] = 'FAILED_AFTER_CUTOVER'
            try:
                runner.command([*rollback_command, 'up', '-d', '--no-build', '--pull', 'never',
                                '--no-deps', *SERVICES], label='rollback-targets')
                rolled_back = inventory(runner)
                restored = target_records(rolled_back)
                if any(restored[s]['image_id'] != old_targets[s]['image_id'] for s in SERVICES):
                    raise UpgradeError('rollback image identity mismatch')
                observed = wait_healthy(runner, restored['api']['id'], old_version, 'rollback')
                outcome.update(rollback='restored_previous_target_images', rollback_version=observed)
            except BaseException as rollback_failure:
                outcome['rollback'] = 'FAILED'
                outcome['rollback_failure_type'] = type(rollback_failure).__name__
        try:
            final = inventory(runner)
            record(evidence / 'containers-final.json', final)
            outcome['non_target_unchanged'] = before is not None and unchanged_others(before, final)
            outcome['non_target_containers'] = len(other_states(final))
        except BaseException:
            outcome['non_target_unchanged'] = None
        record(evidence / 'upgrade.json', outcome)
        raise UpgradeError('upgrade failed; inspect private evidence', evidence=evidence) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--previous-release', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--expected-version', default='0.2.1')
    parser.add_argument('--font-manifest', help='Explicit container path, for example /fonts/manifest.json')
    arguments = parser.parse_args(argv)
    try:
        result = upgrade(arguments.release, arguments.previous_release, arguments.image,
                         expected_version=arguments.expected_version, font_manifest=arguments.font_manifest)
    except (UpgradeError, OSError) as failure:
        print(json.dumps({'status': 'FAILED', 'evidence': getattr(failure, 'evidence', None),
                          'message': str(failure) if isinstance(failure, UpgradeError) else 'invalid release path'}))
        return 1
    print(json.dumps({'status': result['status'], 'version': result['observed_version'],
                      'non_target_unchanged': result['non_target_unchanged'], 'evidence': result['evidence']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
