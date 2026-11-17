"""Check shot.dat from vti2d_FD_RTM kind=1 on the isotropic 2000 m/s model.

kind=1 calls mute_directwave before writing the file, so the direct arrival
is zero. A pass is: that geometric window is zero, nothing arrives before it,
and the strongest sample on the test trace sits just after the window.
Prints OK or FAIL. Same script on the AMD and NVIDIA binaries.
"""

import math
import struct
import sys
from pathlib import Path

NT = 2001
NX = 201
NS = 5
DT = 0.001
DX = 10.0
DZ = 10.0
FAVG = 35.0
VP = 2000.0
EPS = 0.0
FS = NX // NS // 2  # 20
DS = NX // NS       # 40
ZS = 1
TT = 100
SHOT = 0            # is = 1
IX = 180


def mute_bounds(ix, shot):
    mu_x = DX * abs(ix - FS - shot * DS)
    mu_z = DZ * ZS
    mu_t0 = math.hypot(mu_x, mu_z) / (VP * math.sqrt(1.0 + 2.0 * EPS))
    mu_t = int(2.0 / (DT * FAVG))
    t0 = int(mu_t0 / DT)
    # kernel: it > t0 - tt and it < t0 + mu_t + tt
    return t0 - TT, t0 + mu_t + TT


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "shot.dat")
    data = path.read_bytes()
    n = NS * NT * NX
    if len(data) != n * 4:
        print(f"FAIL: {path} is {len(data)} bytes, expected {n * 4}")
        return 1
    samples = struct.unpack("<" + "f" * n, data)
    if not all(math.isfinite(v) for v in samples):
        print("FAIL: shot contains a non-finite sample")
        return 1

    base = SHOT * NT * NX + IX * NT
    trace = samples[base:base + NT]
    lo, hi = mute_bounds(IX, SHOT)
    pre = max(abs(v) for v in trace[: lo + 1])
    muted = max(abs(v) for v in trace[lo + 1 : hi])
    post = trace[hi:hi + 40]
    post_max = max(abs(v) for v in post)
    peak = max(range(NT), key=lambda i: abs(trace[i]))

    ok = (
        muted == 0.0
        and post_max > 0.0
        and pre < 0.02 * post_max
        and hi <= peak < hi + 40
    )
    print(
        f"trace {IX} shot {SHOT + 1}: mute {lo + 1}..{hi - 1}, "
        f"peak {peak * DT:.3f} s, mute ends {hi * DT:.3f} s"
    )
    if ok:
        print("OK")
        return 0
    print(
        f"FAIL: muted max {muted}, before-mute max {pre}, "
        f"after-mute max {post_max}, peak sample {peak}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
