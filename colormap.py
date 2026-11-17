"""Write colormap images for the kind=1 outputs.

    python3 colormap.py

Reads shot.dat (pressure at the receivers) and, when present,
illumination.dat (accumulated P^2+Q^2 on the grid). Writes PPM images
next to those files. No extra Python packages.
"""

import math
import struct
from pathlib import Path

NT = 2001
NX = 201
NZ = 201
NS = 5


def load_f32(path, n):
    data = Path(path).read_bytes()
    if len(data) != n * 4:
        raise SystemExit(f"{path} is {len(data)} bytes, expected {n * 4}")
    return struct.unpack("<" + "f" * n, data)


def clip_level(values):
    usable = sorted(abs(v) for v in values if math.isfinite(v) and v != 0.0)
    if not usable:
        return 1.0
    return usable[min(len(usable) - 1, int(0.99 * (len(usable) - 1)))] or 1.0


def seismic(t):
    """t in [-1, 1]: blue, white, red."""
    t = max(-1.0, min(1.0, t))
    if t >= 0.0:
        return (255, int(255 * (1.0 - t)), int(255 * (1.0 - t)))
    return (int(255 * (1.0 + t)), int(255 * (1.0 + t)), 255)


def hot(t):
    """t in [0, 1]: black, red, yellow, white."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        return (int(255 * u), 0, 0)
    u = (t - 0.5) / 0.5
    return (255, int(255 * u), int(255 * u))


def write_ppm(path, width, height, rgb):
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray()
    for r, g, b in rgb:
        body += bytes((r, g, b))
    Path(path).write_bytes(header + body)
    print(f"wrote {path}  {width}x{height}")


def shot_image(samples):
    level = clip_level(samples)
    width = NS * NX
    pixels = []
    for it in range(NT):
        for shot in range(NS):
            for ix in range(NX):
                value = samples[(shot * NX + ix) * NT + it]
                pixels.append(seismic(value / level))
    write_ppm("shot_colormap.ppm", width, NT, pixels)


def illumination_image(samples):
    level = clip_level(samples)
    pixels = []
    for iz in range(NZ):
        for ix in range(NX):
            value = samples[ix * NZ + iz]
            pixels.append(hot(math.log1p(max(value, 0.0)) / math.log1p(level)))
    write_ppm("illumination_colormap.ppm", NX, NZ, pixels)


def main():
    shot = load_f32("shot.dat", NS * NT * NX)
    shot_image(shot)
    illum_path = Path("illumination.dat")
    if illum_path.is_file():
        illumination_image(load_f32(illum_path, NX * NZ))


if __name__ == "__main__":
    main()
