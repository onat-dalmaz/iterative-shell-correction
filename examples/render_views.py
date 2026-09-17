"""Render viewer views (the same ?z=&a=&b=&zoom= specs the viewer accepts) as a
high-resolution PNG gallery: original / A / B at native pixels, gain-matched,
with the per-slice gradient entropy / energy from results/metrics.json.

  python examples/render_views.py --orig volume.npy --out views.png \
      "z=42&a=ring_k9_N8&b=autofocus&zoom=217,327,644,754" "z=58&a=ring_k9_N8&b=ring_k9_N16&zoom=165,275,315,425"
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from urllib.parse import parse_qs
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from metrics import tissue_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default="views.png")
    ap.add_argument("views", nargs="+")
    args = ap.parse_args()
    vols = {"orig": np.load(args.orig, mmap_mode="r")}
    labels = {"orig": "original"}
    lp = os.path.join(args.results, "labels.json")
    lab = json.load(open(lp)) if os.path.exists(lp) else {}
    for f in sorted(glob.glob(os.path.join(args.results, "*.npy"))):
        n = os.path.splitext(os.path.basename(f))[0]
        if "_fmap" in n or "_selected_freq" in n:
            continue
        vols[n] = np.load(f, mmap_mode="r"); labels[n] = lab.get(n, n)
    mp = os.path.join(args.results, "metrics.json")
    metrics = json.load(open(mp)) if os.path.exists(mp) else {}

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(args.views), 3, figsize=(15, 5.1 * len(args.views)))
    axes = np.atleast_2d(axes)
    for r, spec in enumerate(args.views):
        q = parse_qs(spec)
        z = int(q["z"][0]); a = q["a"][0]; b = q["b"][0]
        zoom = [int(float(v)) for v in q.get("zoom", [""])[0].split(",")] if q.get("zoom", [""])[0] else None
        o = np.abs(np.asarray(vols["orig"][z])).astype(np.float64)
        sl = (slice(zoom[0], zoom[1]), slice(zoom[2], zoom[3])) if zoom else (slice(None), slice(None))
        m = tissue_mask(o)
        omean = o[m].mean()
        vmin, vmax = np.percentile(o[sl], [0.5, 99.7])
        for c, k in enumerate(("orig", a, b)):
            img = np.abs(np.asarray(vols[k][z])).astype(np.float64)
            g = omean / img[m].mean() if k != "orig" else 1.0
            ax = axes[r, c]
            ax.imshow((img * g)[sl], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ttl = f"{labels[k]}  (z={z})"
            if k in metrics and "slices" in metrics[k]:
                s = metrics[k]["slices"]
                ttl += f"\nslice grad. entropy {s['R12'][z]:.2f} (×{s['r12'][z]:.2f})   grad. energy {s['R13'][z] * 1e6:.2f}e-6 (×{s['r13'][z]:.2f})"
            ax.set_title(ttl, fontsize=9); ax.axis("off")
        if zoom:
            axes[r, 0].text(0.01, 0.01, f"y {zoom[0]}–{zoom[1]}, x {zoom[2]}–{zoom[3]}", transform=axes[r, 0].transAxes,
                            color="w", fontsize=8, va="bottom", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
    fig.suptitle("Lean viewer views — original / A / B, native pixels, gain-matched to the original", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
