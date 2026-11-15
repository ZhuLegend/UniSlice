"""Losslessly archive only completed experiment trial directories; retain queue metadata."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def inventory(folder):
    result={}
    for trial in sorted(folder.glob('trial-*')):
        assert trial.is_dir() and not trial.is_symlink()
        for path in trial.rglob('*'):
            assert not path.is_symlink()
            if path.is_file():
                result[str(path.relative_to(folder))]={'sha256':sha(path),'bytes':path.stat().st_size}
    return result


def archive(folder):
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['finished'] and not summary['error'] and not summary['remaining_owned_processes']
    assert (folder/'coverage-audit.json').exists()
    target=folder.with_suffix('.tar.gz')
    record=folder/'archive.json'
    if record.exists():
        saved=json.loads(record.read_text())
        assert sha(target)==saved['archive_sha256']
        if not list(folder.glob('trial-*')):
            return saved
        assert inventory(folder)==saved['files']
    else:
        assert not target.exists()
        files=inventory(folder);assert files
        temporary=Path(str(target)+'.partial')
        with tarfile.open(temporary,'w:gz',compresslevel=1) as packed:
            for trial in sorted(folder.glob('trial-*')):
                packed.add(trial,arcname=trial.name,recursive=True)
        verified={}
        with tarfile.open(temporary,'r:gz') as packed:
            for member in packed:
                if member.isdir():continue
                assert member.isfile() and member.name in files
                with packed.extractfile(member) as source:
                    digest=hashlib.file_digest(source,'sha256').hexdigest()
                assert digest==files[member.name]['sha256'] and member.size==files[member.name]['bytes']
                verified[member.name]=files[member.name]
        assert verified==files and inventory(folder)==files
        os.replace(temporary,target)
        saved={'archive_path':str(target),'archive_sha256':sha(target),'files':files,
            'raw_bytes':sum(row['bytes'] for row in files.values()),'archive_bytes':target.stat().st_size,
            'verification':'All regular-file contents and sizes checked against source SHA-256 before removing this completed queue\'s trial directories. Queue metadata retained.'}
        record.write_text(json.dumps(saved,indent=2)+'\n')
    # Only this queue's freshly generated, hash-verified trial directories are removed.
    for trial in folder.glob('trial-*'):
        shutil.rmtree(trial)
    return saved


def restore(folder,trial_name):
    saved=json.loads((folder/'archive.json').read_text())
    target=Path(saved['archive_path']);assert sha(target)==saved['archive_sha256']
    assert trial_name.startswith('trial-') and '/' not in trial_name and not (folder/trial_name).exists()
    with tarfile.open(target,'r:gz') as packed:
        selected=[member for member in packed.getmembers() if member.name==trial_name or member.name.startswith(trial_name+'/')]
        assert selected
        assert all(member.isdir() or member.isfile() and member.name in saved['files'] for member in selected)
        packed.extractall(folder,members=selected,filter='data')
    for path in (folder/trial_name).rglob('*'):
        if path.is_file():assert sha(path)==saved['files'][str(path.relative_to(folder))]['sha256']
