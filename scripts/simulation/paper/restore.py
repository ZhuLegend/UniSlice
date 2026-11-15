"""Restore only the four runtime versions used by the paper into a new checkout."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from run import ROOT


def restore(experiment, output):
    if output.exists():
        raise ValueError('output already exists; preserve it and choose a fresh directory')
    base = ROOT/'resource/simulation/experiments/runtime'
    versions = json.loads((base/'manifest.json').read_text())['experiments']
    version = versions[experiment]
    if Path(version['archive']).name != version['archive']:
        raise ValueError('runtime archive path must be a filename')
    archive = base/version['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != version['archive_sha256']:
        raise ValueError('runtime archive checksum differs')
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(version['files']):
            raise ValueError('runtime archive member list differs')
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\' in member.name or str(name) != member.name:
                raise ValueError('runtime archive contains an unsafe member')
            payload = bundle.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != version['files'][member.name]:
                raise ValueError('runtime archive member checksum differs: '+member.name)
        output.mkdir(parents=True)
        bundle.extractall(output, filter='data')
    scripts = output/'scripts/simulation'
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ['experiments.py', 'calibrate_parallel.py']:
        destination = scripts/name
        if not destination.exists():
            shutil.copy2(ROOT/'scripts/simulation'/name, destination)
    shutil.copytree(ROOT/'scripts/simulation/paper', scripts/'paper',
                    ignore=shutil.ignore_patterns('__pycache__'))
    inputs = output/'resource/simulation/experiments'
    shutil.copytree(ROOT/'resource/simulation/experiments'/experiment, inputs/experiment,
                    ignore=shutil.ignore_patterns('legacy-*', 'opposed_*'))
    config = output/'.cargo/config.toml'
    config.parent.mkdir(parents=True)
    shutil.copy2(ROOT/'resource/simulation/cargo-config.toml', config)
    record = {**version, 'experiment': experiment,
              'scope': 'Measured Rust/Python execution files and parsed Cargo settings retained; documentation/comment/license edits are listed in the runtime manifest. Portable orchestration and selected inputs added. Build new binaries and record their own identity.'}
    (output/'PAPER_RUNTIME.json').write_text(json.dumps(record, indent=2)+'\n')
    return {'experiment': experiment, 'restored': str(output), 'started': False}
