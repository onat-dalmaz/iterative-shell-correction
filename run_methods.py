"""Run the lean methods on a volume and write results/ for the viewer.

  python run_methods.py --orig volume.npy [--tmap kxyz_time.npy] [--out results] \
                        [--methods ring_k7_N8 ring_k9_N8 ring_k9_N16 autofocus tsc_N16]

Methods (all 3D, [z, x, y] complex volumes):
  ring_k7_N8   exact ring_b0_corr.m  (8 shells, fspecial3 kernel 7)
  ring_k9_N8   exact ts_b0_corr.m    (8 shells, kernel 9)   ← Brian's baseline
  ring_k9_N16  exact ts_b0_corr.m    (16 shells, kernel 9)
  autofocus    3D multi-frequency autofocus, |Imag| criterion, smoothed selection
  tsc_N16      time_seg_corr.m (16 time windows, overlap 8, kernel 7) — needs --tmap
Shell/segment times come from data/tkr_profile_brian.npz unless --tmap is given
(autofocus and tsc use the 3D map when available).
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ring_b0_corr import ring_b0_corr_matlab, ring_b0_corr_v2   # noqa: E402
from autofocus import autofocus_3d, time_map_from_profile      # noqa: E402
from time_seg_corr import time_seg_corr                        # noqa: E402
from freq_seg_corr import freq_seg_corr                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "data", "tkr_profile_brian.npz")
LABELS = {
    "ring_k7_N8": "ring_b0_corr.m — 8 shells, kernel 7",
    "ring_k9_N8": "ts_b0_corr.m — 8 shells, kernel 9 (Brian's baseline)",
    "ring_k9_N16": "ts_b0_corr.m — 16 shells, kernel 9",
    "autofocus": "autofocus 3D — |Imag| criterion, 21 freqs ±150 Hz, σ_idx 5",
    "tsc_N16": "time_seg_corr.m — 16 time windows, overlap 8, kernel 7",
    "ring_predict": "ring 8 shells, kernel 9, PREDICT unwrap (temporal, coherence-gated) — retracted in vivo: ghosts high-|B0| edges",
    "mf_freq_seg": "freq_seg_corr.m — Brian's MF selection (21 freqs ±150 Hz, min smoothed Imag), PATCH z34–50 y180–360 x600–768; original elsewhere",
}
MF_PATCH = (slice(34, 50), slice(180, 360), slice(600, 768))      # (z, rows, cols) around the z=42 skin edge — viewer zoom 180,360,600,768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--tmap", default=None, help="3D sample-time map [z,x,y] (seconds)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--methods", nargs="+", default=["ring_k7_N8", "ring_k9_N8", "ring_k9_N16", "autofocus"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vol = np.asarray(np.load(args.orig, mmap_mode="r"))
    prof = np.load(PROFILE)
    labels_path = os.path.join(args.out, "labels.json")
    labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}
    for m in args.methods:
        t0 = time.time()
        print(f"[{m}] ...", flush=True)
        if m.startswith("ring_k"):
            k = int(m.split("_")[1][1:]); N = int(m.split("_N")[1])
            corr, fmap, info = ring_b0_corr_matlab(vol, N, prof["kr"], prof["t_s"], kernelsize=k, kr_edge=1.0)
            np.save(os.path.join(args.out, f"{m}_fmap_matlab.npy"), fmap)
        elif m == "autofocus":
            tmap = np.load(args.tmap) if args.tmap else time_map_from_profile(vol.shape, PROFILE)
            corr, fsel, info = autofocus_3d(vol, tmap, verbose=True)
            np.save(os.path.join(args.out, f"{m}_selected_freq_hz.npy"), fsel)
        elif m == "ring_predict":
            corr, fmap, info = ring_b0_corr_v2(vol, 8, prof["kr"], prof["t_s"], kr_edge=1.0, unwrap="predict",
                                               fmap_method="ls_weighted", matlab_kernelsize=9)
            np.save(os.path.join(args.out, f"{m}_fmap_ls_weighted.npy"), fmap)
        elif m == "mf_freq_seg":
            if not args.tmap:
                print("  mf_freq_seg needs --tmap (3D sample-time map); skipped"); continue
            zs, rows, cols = MF_PATCH
            corr, fmap = freq_seg_corr(vol, np.load(args.tmap), np.linspace(-150, 150, 21), zs, rows, cols, verbose=True)
            np.save(os.path.join(args.out, f"{m}_selected_freq_patch.npy"), fmap)
        elif m.startswith("tsc_N"):
            if not args.tmap:
                print("  tsc needs --tmap (3D sample-time map); skipped"); continue
            N = int(m.split("_N")[1])
            corr, fmap, info = time_seg_corr(vol, np.load(args.tmap), N)
            if fmap is not None:
                np.save(os.path.join(args.out, f"{m}_fmap_firstinterval.npy"), fmap)
        else:
            print(f"  unknown method {m}; skipped"); continue
        np.save(os.path.join(args.out, f"{m}.npy"), np.asarray(corr).astype(np.complex64))
        labels[m] = LABELS.get(m, m)
        with open(labels_path, "w") as fh:
            json.dump(labels, fh, indent=1)
        print(f"  done in {time.time() - t0:.0f}s → {args.out}/{m}.npy", flush=True)


if __name__ == "__main__":
    main()
