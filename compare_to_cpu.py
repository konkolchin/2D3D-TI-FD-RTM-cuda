"""Compare GPU outputs with a stored CPU reference.

Usage:
    python3 compare_to_cpu.py reference_dir result_dir

reference_dir contains manifest.txt. The CPU file is the reference.
The pass test is the one in compare_volumes.py.
"""

import hashlib
import subprocess
import sys
from pathlib import Path


def parse_manifest(path):
    outputs = []
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("output:"):
            outputs.append(line.split(None, 1)[1].strip())
        elif line.startswith("sha256:"):
            name, digest = line.split(None, 2)[1:]
            hashes[name] = digest
    if not outputs:
        raise SystemExit(f"FAIL: {path} lists no outputs")
    return outputs, hashes


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 compare_to_cpu.py reference_dir result_dir")
    reference_dir = Path(sys.argv[1])
    result_dir = Path(sys.argv[2])
    manifest = reference_dir / "manifest.txt"
    outputs, hashes = parse_manifest(manifest)
    compare = Path(__file__).with_name("compare_volumes.py")
    failed = False
    for name in outputs:
        reference = reference_dir / name
        other = result_dir / name
        print(f"== {name}", flush=True)
        digest = file_sha256(reference)
        recorded = hashes.get(name)
        if recorded is None or recorded != digest:
            print(f"FAIL: {reference} sha256 {digest} manifest {recorded}")
            failed = True
            continue
        completed = subprocess.run(
            [sys.executable, str(compare), str(reference), str(other)],
            check=False,
        )
        if completed.returncode != 0:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
