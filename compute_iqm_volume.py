"""Volume-level sharpness measures per method, written into results/metrics.json as
metrics[<method>]["iqm_volume"] for the viewer's "volume" column:

  edge_width   median 10-90 % width over ALL edges of results/edge_widths_neutral.npz
               (edges detected on the uncorrected image; run edge_width.py first)
  limS, TG, AES  mean over sampled slices (every --step) of the per-slice values from
               iqm.compare_iqms inside the mask, plus the ratio to the uncorrected volume

  python compute_iqm_volume.py --orig volume.npy --results results [--mask mask.npy] [--step 3]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import tissue_mask                    # noqa: E402
from iqm import compare_iqms, sigma_air             # noqa: E402
from edge_width import detect_edges                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--mask", default=None); ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    orig = np.load(a.orig, mmap_mode="r"); nz = orig.shape[0]
    mask3 = np.load(a.mask, mmap_mode="r") if a.mask else None
    mp = os.path.join(a.results, "metrics.json")
    metrics = json.load(open(mp)) if os.path.exists(mp) else {}
    ew = None
    ewp = os.path.join(a.results, "edge_widths_neutral.npz")
    if os.path.exists(ewp):
        e = np.load(ewp); ew = {str(k): e["widths"][i] for i, k in enumerate(e["methods"])}
    vols = {"orig": orig}
    for f in sorted(glob.glob(os.path.join(a.results, "*.npy"))):
        n = os.path.splitext(os.path.basename(f))[0]
        if "_fmap" in n or "_selected_freq" in n or "_patch" in n:
            continue
        vols[n] = np.load(f, mmap_mode="r")
    zs = [z for z in range(2, nz, a.step)]
    O, M, PTS, SIG = {}, {}, {}, {}
    for z in zs:
        o = np.abs(np.asarray(orig[z])).astype(np.float64)
        m = np.asarray(mask3[z]).astype(bool) if mask3 is not None else tissue_mask(o)
        if m.sum() < 500:
            continue
        O[z] = o; M[z] = m; PTS[z] = [(x, yc, s) for (_, x, yc, s) in detect_edges(o[None], slice_indices=[0])]; SIG[z] = sigma_air(o, m)
    print(f"{len(O)} slices, {len(vols)} volumes", flush=True)
    for name, v in vols.items():
        acc = {k: [] for k in ("TG", "AES", "TG_orig", "AES_orig", "limS", "limS_orig", "edge_width", "edge_width_orig")}
        for z in O:
            q = compare_iqms(O[z], np.abs(np.asarray(v[z])).astype(np.float64), M[z], pts=PTS[z], sig=SIG[z])
            for k in acc:
                acc[k].append(q[k])
        r = {k: float(np.nanmean(acc[k])) if k.startswith(("TG", "AES")) else float(np.nanmedian(acc[k])) for k in acc}
        out = {"TG": r["TG"], "AES": r["AES"], "limS": r["limS"], "edge_width_slices": r["edge_width"],
               "TG_ratio": r["TG"] / r["TG_orig"], "AES_ratio": r["AES"] / r["AES_orig"], "limS_ratio": r["limS"] / r["limS_orig"],
               "n_slices": len(O)}
        if ew is not None and name in ew:
            out["edge_width"] = float(np.nanmedian(ew[name])); out["n_edges"] = int(np.isfinite(ew[name]).sum())
        else:
            out["edge_width"] = r["edge_width"]
        metrics.setdefault(name, {})["iqm_volume"] = out
        print(f"  {name:24s} edge {out['edge_width']:.3f} px  limS {out['limS']:.1f}  TG x{out['TG_ratio']:.3f}  AES x{out['AES_ratio']:.3f}", flush=True)
    json.dump(metrics, open(mp, "w"), indent=1)
    print(f"wrote {mp}")


if __name__ == "__main__":
    main()
