"""Compare two raw little-endian float32 volumes.

Usage:
    python3 compare_volumes.py reference.dat other.dat

The first file is the reference. Prints max and RMS differences.
OK means the largest absolute difference is within 1% of the larger peak.
The files are not expected to be bit-identical.
"""

import array
import math
import sys
from pathlib import Path

REL_OK = 0.01


def load(path):
    data = Path(path).read_bytes()
    if len(data) == 0 or len(data) % 4 != 0:
        raise SystemExit(f"FAIL: {path} is {len(data)} bytes, not a whole number of floats")
    samples = array.array("f")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    for value in samples:
        if not math.isfinite(value):
            raise SystemExit(f"FAIL: {path} contains a non-finite sample")
    return samples


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 compare_volumes.py reference.dat other.dat")
    ref = load(sys.argv[1])
    other = load(sys.argv[2])
    if len(ref) != len(other):
        raise SystemExit(f"FAIL: sample count {len(ref)} vs {len(other)}")
    n = len(ref)
    peak = 0.0
    max_abs = 0.0
    sum_sq = 0.0
    n_differ = 0
    for a, b in zip(ref, other):
        peak = max(peak, abs(a), abs(b))
        diff = a - b
        max_abs = max(max_abs, abs(diff))
        sum_sq += diff * diff
        if a != b:
            n_differ += 1
    if peak == 0.0:
        print("FAIL: both volumes are zero")
        return 1
    rms = math.sqrt(sum_sq / n)
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
