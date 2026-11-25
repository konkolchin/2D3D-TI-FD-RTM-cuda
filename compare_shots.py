"""Compare two shot.dat files from vti2d_FD_RTM kind=1.

Usage:
    python3 compare_shots.py shot_nvidia.dat shot_amd.dat

The first file is the reference. Prints max and RMS differences.
OK means the largest absolute difference is within 1% of the larger peak.
The files are not expected to be bit-identical.
The shot count is the file length divided by one shot (NT by NX floats).
"""

import math
import struct
import sys
from pathlib import Path

NT = 2001
NX = 201
TRACE = NT * NX
REL_OK = 0.01


def load(path):
    data = Path(path).read_bytes()
    if len(data) == 0 or len(data) % (TRACE * 4) != 0:
        raise SystemExit(
            f"FAIL: {path} is {len(data)} bytes, not a whole number of "
            f"{TRACE}-sample shots"
        )
    ns = len(data) // (TRACE * 4)
    samples = struct.unpack("<" + "f" * (ns * TRACE), data)
    if not all(math.isfinite(v) for v in samples):
        raise SystemExit(f"FAIL: {path} contains a non-finite sample")
    return ns, samples


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 compare_shots.py shot_nvidia.dat shot_amd.dat")
    ns_ref, ref = load(sys.argv[1])
    ns_other, other = load(sys.argv[2])
    if ns_ref != ns_other or len(ref) != len(other):
        raise SystemExit(f"FAIL: shot count {ns_ref} vs {ns_other}")
    n = len(ref)
    peak = max(max(abs(v) for v in ref), max(abs(v) for v in other))
    if peak == 0.0:
        print("FAIL: both shots are zero")
        return 1
    diff = [a - b for a, b in zip(ref, other)]
    max_abs = max(abs(v) for v in diff)
    rms = math.sqrt(sum(v * v for v in diff) / n)
    n_differ = sum(a != b for a, b in zip(ref, other))
    print(f"shots {ns_ref}")
    print(f"samples {n}")
    print(f"peak {peak:.6g}")
    print(f"max abs diff {max_abs:.6g}  ({100.0 * max_abs / peak:.4f}% of peak)")
    print(f"rms diff {rms:.6g}  ({100.0 * rms / peak:.4f}% of peak)")
    print(f"samples that differ {n_differ}")
    if max_abs <= REL_OK * peak:
        print("OK")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())