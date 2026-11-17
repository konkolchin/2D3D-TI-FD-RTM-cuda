"""Compare two shot.dat files from vti2d_FD_RTM kind=1.

Usage:
    python3 compare_shots.py shot_nvidia.dat shot_amd.dat

The first file is the reference. Prints max and RMS differences.
OK means the largest absolute difference is within 1% of the larger peak.
The files are not expected to be bit-identical.
"""

import math
import struct
import sys
from pathlib import Path

NS = 5
NT = 2001
NX = 201
N = NS * NT * NX
REL_OK = 0.01


def load(path):
    data = Path(path).read_bytes()
    if len(data) != N * 4:
        raise SystemExit(f"FAIL: {path} is {len(data)} bytes, expected {N * 4}")
    samples = struct.unpack("<" + "f" * N, data)
    if not all(math.isfinite(v) for v in samples):
        raise SystemExit(f"FAIL: {path} contains a non-finite sample")
    return samples


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 compare_shots.py shot_nvidia.dat shot_amd.dat")
    ref = load(sys.argv[1])
    other = load(sys.argv[2])
    peak = max(max(abs(v) for v in ref), max(abs(v) for v in other))
    if peak == 0.0:
        print("FAIL: both shots are zero")
        return 1
    diff = [a - b for a, b in zip(ref, other)]
    max_abs = max(abs(v) for v in diff)
    rms = math.sqrt(sum(v * v for v in diff) / N)
    n_differ = sum(a != b for a, b in zip(ref, other))
    print(f"samples {N}")
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
