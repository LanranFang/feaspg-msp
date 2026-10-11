"""Verify every file distributed with the artifact."""
import hashlib
from . import ROOT


def main():
    failures=[];count=0
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split('  ',1);path=ROOT/name;count+=1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append(name)
    if failures:
        raise RuntimeError('Missing or changed artifact files: '+', '.join(failures))
    print(f'Verified {count} distributed files. Generated output files are not part of this manifest.')


if __name__=='__main__':main()
