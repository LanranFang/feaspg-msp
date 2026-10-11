"""Build a clean GitHub upload archive and refresh distributed-file checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from . import ROOT


def distributed(path):
    rel = path.relative_to(ROOT)
    if rel.parts[0] == 'paper' or path.suffix.lower() in (
        '.pdf', '.png', '.jpg', '.jpeg', '.svg', '.eps', '.webp', '.gif', '.tif', '.tiff'
    ):
        return False
    if not path.is_file() or any(p in rel.parts for p in (
        '__pycache__', '.pytest_cache', '.git', '.venv', 'node_modules'
    )):
        return False
    if path.suffix in ('.pyc', '.so', '.log', '.pkl') or any(
        p.endswith('.egg-info') for p in rel.parts
    ):
        return False
    if rel.parts[0] == 'outputs':
        return False
    if rel.parts[0] not in ('reference_results', 'reference_checkpoints') and 'outputs' in rel.parts:
        if rel.parts[0] != 'synthetic_hydro_certificate':
            return False
        if len(rel.parts) > 3 and rel.parts[2] == 'scaling_v2' and rel.parts[3].startswith('r'):
            return path.name in ('fit.npz', 'protocol.json', 'reference.json',
                                 'fitting.json', 'continuous_audits.json', 'results.csv')
        return len(rel.parts) > 3 and rel.parts[2:4] == ('certificate_v1', 'frozen')
    return True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'feaspg-msp-github.zip')
    args = parser.parse_args()
    archive = args.output.resolve()
    if archive.is_relative_to(ROOT):
        parser.error('Write the archive outside the repository to avoid including it in itself.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    provenance = ROOT / 'artifact/source_provenance.json'
    old = {r['file']: r for r in json.loads(provenance.read_text())}
    exports = json.loads((ROOT / 'artifact/preprint_provenance.json').read_text())['new_exports']
    for entry in exports:
        old[entry['destination']] = {
            'file': entry['destination'], 'source': entry['source'],
            'source_sha256': entry.get('source_sha256', entry.get('sha256')),
            'packaging_change': entry.get('change', 'Unchanged source snapshot.'),
        }
    records = []
    for path in sorted(ROOT.rglob('*')):
        if not distributed(path) or path == provenance or path.name == 'MANIFEST.sha256':
            continue
        name = path.relative_to(ROOT).as_posix()
        if name in old:
            record = old[name]
            record['packaged_sha256'] = sha(path)
            records.append(record)
    provenance.write_text(json.dumps(records, indent=2) + '\n')
    files = [p for p in sorted(ROOT.rglob('*')) if distributed(p) and p.name != 'MANIFEST.sha256']
    manifest = ROOT / 'MANIFEST.sha256'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
    files.append(manifest)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            info = zipfile.ZipInfo(ROOT.name + '/' + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes(), compresslevel=9)
        if z.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed.')
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'files': len(files),
                      'bytes': archive.stat().st_size, 'sha256': sha(archive)}, indent=2))


if __name__ == '__main__':
    main()
