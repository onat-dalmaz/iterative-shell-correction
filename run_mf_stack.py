"""Brian's multifrequency reconstruction "to 4D": run freq_corr.m — exactly,
in full 3D — at every frequency on a grid and save each volume, so the
viewer's frequency slider just indexes the stack (manual per-patch tuning =
the oracle Brian describes in his log).

  python run_mf_stack.py --orig volume.npy --tmap kxyz_time.npy [--fmin -150 --fmax 150 --step 10] [--complex]

Writes results/mf_stack/f{+ddd}.npy (|image| float32 by default; --complex
keeps complex64 = 2× disk) and results/mf_stack/freqs.json.
Each volume is  ift3( ft3(image) .* exp(2πi·kxyz_time·f) )  — freq_corr.m
with skip_fwd_fft after one forward transform (bit-identical to calling
freq_corr per frequency).
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from time_seg_corr import ft3, ift3   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--tmap", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "mf_stack"))
    ap.add_argument("--fmin", type=float, default=-150); ap.add_argument("--fmax", type=float, default=150)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--complex", action="store_true", help="store complex64 instead of |image| float32")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    freqs = np.arange(args.fmin, args.fmax + 1e-6, args.step)
    vol = np.asarray(np.load(args.orig, mmap_mode="r"))
    tmap = np.load(args.tmap)
    xyz = lambda a: np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
    zxy = lambda a: np.ascontiguousarray(np.transpose(a, (2, 0, 1)))
    t0 = time.time()
    K = ft3(xyz(vol).astype(np.complex64)).astype(np.complex64)          # ft3 once (freq_corr with skip_fwd_fft=1)
    t = xyz(tmap).astype(np.float32)
    del vol, tmap
    print(f"k-space ready ({time.time() - t0:.0f}s); {len(freqs)} frequencies {args.fmin:+.0f}..{args.fmax:+.0f} step {args.step:g} Hz", flush=True)
    names = []
    for f in freqs:
        t1 = time.time()
        img = ift3(K * np.exp(2j * np.pi * t * np.float32(f)).astype(np.complex64))          # freq_corr.m
        name = f"f{f:+04.0f}.npy"
        np.save(os.path.join(args.out, name), zxy(img).astype(np.complex64) if args.complex else np.abs(zxy(img)).astype(np.float32))
        names.append(name); del img
        print(f"  {f:+6.0f} Hz → {name}  ({time.time() - t1:.1f}s)", flush=True)
    with open(os.path.join(args.out, "freqs.json"), "w") as fh:
        json.dump({"freqs_hz": freqs.tolist(), "files": names, "dtype": "complex64" if args.complex else "float32 magnitude",
                   "formula": "ift3(ft3(image) .* exp(2*pi*i*kxyz_time*f))  [freq_corr.m]"}, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
