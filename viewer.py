"""Lean slice viewer: original vs two methods with synchronised zoom/pan, and the
sharpness measures for the whole volume, the slice and the zoomed window.

  python viewer.py --orig volume.npy [--results results] [--mask mask.npy] [--port 8053]

results/ holds the method volumes (<name>.npy + labels.json, labels_short.json),
metrics.json (compute_metrics.py + compute_iqm_volume.py) and, optionally,
edge_widths_neutral.npz (edge_width.py) and mf_stack/ (run_mf_stack.py).

Measures (all inside the mask, candidates gain-matched to the original):
  edge width   median 10-90 % width (px) of profiles at strong edges detected on the
               UNCORRECTED image (lower = sharper) — the sharpness measure
  Lim S        CNR / (20-80 % distance) on the same edges, Lim et al. 2019 (higher = sharper)
  TG, AES      Tenengrad, average edge strength at Canny edges (Marchetto & Eichhorn 2024),
               shown as ratio to the original (AES tracks visual scores best)
  R12, R13     gradient entropy / normalised gradient energy: gradient CONCENTRATION
               (artifact reading, not sharpness), ratio to the original
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import tissue_mask, r12_r13, ratios  # noqa: E402
from iqm import compare_iqms, sigma_air            # noqa: E402
from edge_width import detect_edges                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
ap.add_argument("--port", type=int, default=8053)
ap.add_argument("--mask", default=None, help="bool volume [z,rows,cols] used for the metrics instead of the per-slice Otsu mask (e.g. a breast mask)")
ARGS = ap.parse_args()
MASK3 = np.load(ARGS.mask, mmap_mode="r") if ARGS.mask else None

VOLS: dict[str, np.ndarray] = {"orig": np.load(ARGS.orig, mmap_mode="r")}
LABELS: dict[str, str] = {"orig": "original (uncorrected)"}
_lab_path = os.path.join(ARGS.results, "labels.json")
_labels = json.load(open(_lab_path)) if os.path.exists(_lab_path) else {}
_short_path = os.path.join(ARGS.results, "labels_short.json")
SHORT: dict[str, str] = json.load(open(_short_path)) if os.path.exists(_short_path) else {}
for f in sorted(glob.glob(os.path.join(ARGS.results, "*.npy"))):
    name = os.path.splitext(os.path.basename(f))[0]
    if "_fmap" in name or "_selected_freq" in name or "_patch" in name:
        continue
    VOLS[name] = np.load(f, mmap_mode="r"); LABELS[name] = _labels.get(name, name)
# Brian's multifrequency stack (run_mf_stack.py): one |freq_corr| volume per frequency.
MF_DIR = os.path.join(ARGS.results, "mf_stack")
MF_FREQS: list[float] = []; MF_VOLS: list[np.ndarray] = []
if os.path.exists(os.path.join(MF_DIR, "freqs.json")):
    _mf = json.load(open(os.path.join(MF_DIR, "freqs.json")))
    for f, n in zip(_mf["freqs_hz"], _mf["files"]):
        p = os.path.join(MF_DIR, n)
        if os.path.exists(p):
            MF_FREQS.append(float(f)); MF_VOLS.append(np.load(p, mmap_mode="r"))
if MF_FREQS:
    VOLS["mf_stack"] = None      # placeholder: resolved per (f, z) in get_slice
    LABELS["mf_stack"] = f"freq_corr.m at the slider frequency (Brian's MF recon, {len(MF_FREQS)} freqs {MF_FREQS[0]:+.0f}..{MF_FREQS[-1]:+.0f} Hz)"
METHODS = list(VOLS.keys())
NZ, NY, NX = VOLS["orig"].shape
_metrics_path = os.path.join(ARGS.results, "metrics.json")
METRICS = json.load(open(_metrics_path)) if os.path.exists(_metrics_path) else {}


def short(name: str) -> str:
    s = SHORT.get(name)
    if s:
        return s
    lab = LABELS.get(name, name)
    return lab if len(lab) <= 44 else lab[:42] + "…"


def mf_index(f_hz: float) -> int:
    return int(np.argmin(np.abs(np.asarray(MF_FREQS) - f_hz)))


def get_slice(method: str, z: int, f_hz: float = 0.0) -> np.ndarray:
    """|image| slice [rows, cols] for any method, including the MF stack at the slider frequency."""
    if method == "mf_stack":
        return np.abs(np.asarray(MF_VOLS[mf_index(f_hz)][z])).astype(np.float32)
    return np.abs(np.asarray(VOLS[method][z])).astype(np.float32)


_MASK: dict[int, np.ndarray] = {}
_BASE: dict[int, dict] = {}
_SLICE: dict[tuple, dict] = {}
_EDGES: dict[int, list] = {}
_SIG: dict[int, float] = {}
_IQM: dict[tuple, dict] = {}


def mask_for(z):
    if z not in _MASK:
        _MASK[z] = np.asarray(MASK3[z]).astype(bool) if MASK3 is not None else tissue_mask(np.asarray(VOLS["orig"][z]))
    return _MASK[z]


def _key(method, z, win, f_hz):
    return (method, z, tuple(win) if win is not None else None, mf_index(f_hz) if method == "mf_stack" else None)


def slice_r12r13(method, z, win, f_hz=0.0):
    """R12/R13 of the slice (win=None) or of the window, with ratios vs the original."""
    key = ("r", *_key(method, z, win, f_hz))
    if key in _SLICE:
        return _SLICE[key]
    m = mask_for(z); o = get_slice("orig", z); c = get_slice(method, z, f_hz)
    if win is not None:
        y0, y1, x0, x1 = win; m = m[y0:y1, x0:x1]; o = o[y0:y1, x0:x1]; c = c[y0:y1, x0:x1]
    base = r12_r13(o, m); cand = r12_r13(c, m)
    out = {"R12": cand["R12"], "R13": cand["R13"], "R12_orig": base["R12"], "R13_orig": base["R13"], **ratios(base, cand)}
    _SLICE[key] = out
    return out


def iqms_for(method, z, win, f_hz=0.0):
    """Edge width, Lim S, TG, AES (+NGS/GE/IE) of a method vs the original, per slice or per window."""
    key = _key(method, z, win, f_hz)
    if key in _IQM:
        return _IQM[key]
    o = get_slice("orig", z); m = mask_for(z)
    if z not in _EDGES:
        _EDGES[z] = [(x, yc, s) for (_, x, yc, s) in detect_edges(o[None], slice_indices=[0])]
        _SIG[z] = sigma_air(o, m)
    out = compare_iqms(o, get_slice(method, z, f_hz), m, box=(tuple(win) if win is not None else None), pts=_EDGES[z], sig=_SIG[z])
    _IQM[key] = out
    return out


def make_fig(arr2d, z, show_mask, zoom):
    a = np.abs(arr2d).astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.7])
    fig = go.Figure(go.Heatmap(z=a, colorscale="Gray", zmin=lo, zmax=hi, showscale=False,
                               hovertemplate="x=%{x} y=%{y} |I|=%{z:.3g}<extra></extra>"))
    fig.update_layout(margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="black", plot_bgcolor="black",
                      uirevision="keep", dragmode="pan")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1.0, constrain="domain")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain", autorange="reversed")
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        fig.update_xaxes(range=[x0, x1]); fig.update_yaxes(range=[y1, y0], autorange=False)
    if show_mask:
        m = mask_for(z); edge = m & ~binary_erosion(m)
        yi, xi = np.where(edge)
        if len(xi) > 4000:
            sel = np.random.default_rng(0).choice(len(xi), 4000, replace=False); xi, yi = xi[sel], yi[sel]
        fig.add_scatter(x=xi, y=yi, mode="markers", marker=dict(size=1.4, color="#7CFC00"), hoverinfo="skip", showlegend=False)
    return fig


app = Dash(__name__)
app.title = "cones off-resonance viewer (lean)"
LAB = {"fontSize": "12px", "fontWeight": "600", "color": "#333", "marginBottom": "2px"}
TITLE = {"margin": "4px 0", "fontSize": "14px", "height": "20px", "lineHeight": "20px", "overflow": "hidden",
         "whiteSpace": "nowrap", "textOverflow": "ellipsis"}
_mf_marks = {f: (f"{f:+.0f}" if abs(f) % 50 == 0 else "") for f in MF_FREQS} if MF_FREQS else {f: f"{f:+.0f}" for f in range(-150, 151, 50)}
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="view"),
    html.H3("Cones off-resonance correction — lean viewer", style={"marginBottom": "2px"}),
    html.Div(f"Measures inside the {'supplied mask (' + os.path.basename(ARGS.mask) + ')' if ARGS.mask else 'Otsu tissue mask'}: "
             "edge width = median 10–90 % width (px) at strong edges detected on the uncorrected image (lower = sharper; the sharpness measure); "
             "Lim S = CNR / (20–80 % distance) on the same edges (higher = sharper); TG, AES = Tenengrad and average edge strength, ratio vs original "
             "(AES tracks visual scores best; TG also rewards high-k ripple); R12, R13 = gradient entropy / normalised gradient energy (artifact reading, not sharpness). "
             "Zoom or pan in any panel: the others follow and the 'window' column is computed on the visible region.",
             style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}),
    html.Div([
        html.Div([html.Div("Method A", style=LAB),
                  dcc.Dropdown(id="a-sel", options=[{"label": short(k), "value": k, "title": LABELS[k]} for k in METHODS],
                               value=METHODS[1] if len(METHODS) > 1 else "orig", clearable=False, style={"width": "340px"})]),
        html.Div([html.Div("Method B", style=LAB),
                  dcc.Dropdown(id="b-sel", options=[{"label": short(k), "value": k, "title": LABELS[k]} for k in METHODS],
                               value=METHODS[2] if len(METHODS) > 2 else "orig", clearable=False, style={"width": "340px"})]),
        html.Div([html.Div("Zoom box  y0,y1,x0,x1  (blank = full)", style=LAB),
                  dcc.Input(id="zoom-box", type="text", value="", debounce=True, style={"width": "200px"})]),
        html.Div([dcc.Checklist(id="show-mask", options=[{"label": "  mask outline", "value": "yes"}], value=[])]),
    ], style={"display": "flex", "gap": "14px", "alignItems": "flex-end", "flexWrap": "wrap", "marginBottom": "8px"}),
    html.Div([html.Div("MF frequency (Hz) — select 'freq_corr at slider frequency' as A or B"
                       + ("" if MF_FREQS else "  [no stack: run run_mf_stack.py]"), style=LAB),
              dcc.Slider(id="f-sel", min=MF_FREQS[0] if MF_FREQS else -150, max=MF_FREQS[-1] if MF_FREQS else 150,
                         value=0, step=None if MF_FREQS else 10, marks=_mf_marks,
                         tooltip={"placement": "bottom", "always_visible": True}),
              html.Button("scan f in window", id="scan-btn", n_clicks=0, style={"marginTop": "4px"}),
              html.Div(id="scan-out", style={"fontFamily": "ui-monospace, monospace", "fontSize": "12px", "whiteSpace": "pre"})],
             style={"marginBottom": "8px"}),
    html.Div([html.Div("Slice", style=LAB),
              dcc.Slider(id="z-sel", min=0, max=NZ - 1, value=NZ // 2, step=1,
                         marks={i: str(i) for i in range(0, NZ, max(1, NZ // 8))},
                         tooltip={"placement": "bottom", "always_visible": True})], style={"marginBottom": "8px"}),
    html.Div([
        html.Div([html.H4(id="t-orig", style=TITLE), dcc.Graph(id="g-orig", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1", "minWidth": 0}),
        html.Div([html.H4(id="t-a", style=TITLE), dcc.Graph(id="g-a", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1", "minWidth": 0}),
        html.Div([html.H4(id="t-b", style=TITLE), dcc.Graph(id="g-b", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1", "minWidth": 0}),
    ], style={"display": "flex", "gap": "8px"}),
    html.Div(id="_sync", style={"display": "none"}),
    html.Div(id="metrics", style={"marginTop": "10px"}),
], style={"padding": "10px 16px", "fontFamily": "system-ui, sans-serif"})


@app.callback(Output("a-sel", "value"), Output("b-sel", "value"), Output("z-sel", "value"), Output("zoom-box", "value"),
              Output("f-sel", "value"), Input("url", "search"))
def from_url(search):
    """Shareable views: ?z=42&a=ring_k9_N8&b=mf_stack&f=-60&zoom=217,327,644,754"""
    from urllib.parse import parse_qs
    q = parse_qs((search or "").lstrip("?"))
    a = q.get("a", [METHODS[1] if len(METHODS) > 1 else "orig"])[0]
    b = q.get("b", [METHODS[2] if len(METHODS) > 2 else "orig"])[0]
    z = int(q.get("z", [NZ // 2])[0])
    zoom = q.get("zoom", [""])[0]
    f = float(q.get("f", [0])[0])
    if MF_FREQS:
        f = MF_FREQS[mf_index(f)]
    return (a if a in METHODS else METHODS[0], b if b in METHODS else METHODS[0], max(0, min(NZ - 1, z)), zoom, f)


def parse_zoom(s):
    try:
        v = [int(float(x)) for x in s.replace(";", ",").split(",")]
        return tuple(v) if len(v) == 4 else None
    except Exception:
        return None


def _clip_win(win):
    if win is None:
        return None
    y0, y1, x0, x1 = win
    y0, y1 = max(0, min(NY, int(y0))), max(0, min(NY, int(y1))); x0, x1 = max(0, min(NX, int(x0))), max(0, min(NX, int(x1)))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    return (y0, y1, x0, x1)


def _title(method, f):
    if method == "mf_stack":
        return f"freq_corr at {MF_FREQS[mf_index(f)]:+.0f} Hz (MF recon)"
    return short(method)


@app.callback(Output("g-orig", "figure"), Output("g-a", "figure"), Output("g-b", "figure"),
              Output("t-orig", "children"), Output("t-a", "children"), Output("t-b", "children"),
              Output("t-a", "title"), Output("t-b", "title"),
              Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"),
              Input("show-mask", "value"), Input("zoom-box", "value"), Input("f-sel", "value"))
def update(a, b, z, show_mask, zoom_s, f):
    z = int(z); show = bool(show_mask); zoom = _clip_win(parse_zoom(zoom_s or "")); f = float(f or 0)
    return (make_fig(get_slice("orig", z), z, show, zoom), make_fig(get_slice(a, z, f), z, show, zoom),
            make_fig(get_slice(b, z, f), z, show, zoom),
            f"original  (z={z})", f"A: {_title(a, f)}", f"B: {_title(b, f)}", LABELS.get(a, a), LABELS.get(b, b))


# ---------------------------------------------------------------- metrics tables
def _fmt(v, nd=2, sci=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}e}" if sci else f"{v:.{nd}f}"


ROWS = [("edge width (px)", "ew"), ("Lim S", "lims"), ("TG (×orig)", "tg"), ("AES (×orig)", "aes"),
        ("R12 (×orig)", "r12"), ("R13 (×orig)", "r13")]


def vol_cells(method):
    """Volume column: precomputed (compute_metrics.py + compute_iqm_volume.py)."""
    v = METRICS.get(method, {}).get("volume"); q = METRICS.get(method, {}).get("iqm_volume")
    if method == "mf_stack" or (v is None and q is None):
        return {k: "–" for _, k in ROWS}, None
    c = {}
    c["ew"] = _fmt(q.get("edge_width")) if q else "–"
    c["lims"] = _fmt(q.get("limS"), 1) if q else "–"
    c["tg"] = _fmt(q.get("TG_ratio"), 3) if q else "–"
    c["aes"] = _fmt(q.get("AES_ratio"), 3) if q else "–"
    c["r12"] = _fmt(v["r12"], 3) if v else "–"
    c["r13"] = _fmt(v["r13"], 3) if v else "–"
    n = q.get("n_edges") if q else None
    return c, n


def local_cells(method, z, win, f):
    q = iqms_for(method, z, win, f); r = slice_r12r13(method, z, win, f)
    if method == "orig":
        return {"ew": _fmt(q["edge_width_orig"]), "lims": _fmt(q["limS_orig"], 1), "tg": "1.000", "aes": "1.000", "r12": "1.000", "r13": "1.000"}, q["n_edges"]
    return {"ew": _fmt(q["edge_width"]), "lims": _fmt(q["limS"], 1), "tg": _fmt(q["TG_ratio"], 3), "aes": _fmt(q["AES_ratio"], 3),
            "r12": _fmt(r["r12"], 3), "r13": _fmt(r["r13"], 3)}, q["n_edges"]


TD = {"padding": "3px 10px", "textAlign": "right", "fontVariantNumeric": "tabular-nums", "borderBottom": "1px solid #eee"}
TH = {"padding": "4px 10px", "textAlign": "right", "fontWeight": "600", "borderBottom": "2px solid #ccc", "whiteSpace": "nowrap"}
COL = {"orig": "#444", "A": "#1E88E5", "B": "#E65100"}


def scope_table(title, sub, cells, n_edges):
    """cells: {'orig': {...}, 'A': {...}, 'B': {...}}"""
    head = html.Tr([html.Th(title, style={**TH, "textAlign": "left"})] + [html.Th(k, style={**TH, "color": COL[k]}) for k in ("orig", "A", "B")])
    body = [html.Tr([html.Td(lab, style={**TD, "textAlign": "left", "color": "#333"})] + [html.Td(cells[k][key], style=TD) for k in ("orig", "A", "B")])
            for lab, key in ROWS]
    foot = html.Div((sub + (f" · {n_edges} edges" if n_edges is not None else "")), style={"fontSize": "11px", "color": "#888", "padding": "2px 10px"})
    return html.Div([html.Table([html.Thead(head), html.Tbody(body)], style={"borderCollapse": "collapse", "fontSize": "13px", "width": "100%"}), foot],
                    style={"flex": "1", "minWidth": "300px", "background": "#fafafa", "border": "1px solid #ddd", "borderRadius": "8px", "padding": "6px 4px"})


@app.callback(Output("metrics", "children"), Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"),
              Input("zoom-box", "value"), Input("f-sel", "value"), Input("view", "data"))
def update_metrics(a, b, z, zoom_s, f, view):
    z = int(z); f = float(f or 0)
    win = _clip_win(tuple(view)) if view else _clip_win(parse_zoom(zoom_s or ""))
    legend = html.Div([html.Span("A ", style={"color": COL["A"], "fontWeight": "700"}), html.Span(_title(a, f) + "    "),
                       html.Span("B ", style={"color": COL["B"], "fontWeight": "700"}), html.Span(_title(b, f))],
                      style={"fontSize": "13px", "marginBottom": "6px"})
    vo, _ = vol_cells("orig"); va, na = vol_cells(a); vb, _ = vol_cells(b)
    so, ns = local_cells("orig", z, None, f); sa, _ = local_cells(a, z, None, f); sb, _ = local_cells(b, z, None, f)
    tables = [scope_table("volume", "whole volume (breast mask); edge width over all sampled edges", {"orig": vo, "A": va, "B": vb}, na),
              scope_table(f"slice z={z}", "this slice", {"orig": so, "A": sa, "B": sb}, ns)]
    if win is not None:
        wo, nw = local_cells("orig", z, win, f); wa, _ = local_cells(a, z, win, f); wb, _ = local_cells(b, z, win, f)
        tables.append(scope_table("window", f"rows {win[0]}–{win[1]}, cols {win[2]}–{win[3]} on z={z}; few edges → ±0.5 px noise; "
                                            "on line-like structures (skin band) the 10–90 % width is undefined — trust slice/volume rows",
                                  {"orig": wo, "A": wa, "B": wb}, nw))
    else:
        tables.append(html.Div("window: zoom into a panel (scroll / drag) or type a zoom box",
                               style={"flex": "1", "minWidth": "300px", "color": "#999", "padding": "20px 10px", "fontSize": "13px"}))
    return html.Div([legend, html.Div(tables, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})])


@app.callback(Output("scan-out", "children"), Input("scan-btn", "n_clicks"), Input("z-sel", "value"), Input("zoom-box", "value"),
              Input("view", "data"), prevent_initial_call=True)
def scan_box(n, z, zoom_s, view):
    """Evaluate every stack frequency inside the window: edge width, AES and R13 vs the original —
    an aid for the manual tuning, not an oracle."""
    from dash import ctx
    if not MF_FREQS or ctx.triggered_id != "scan-btn":
        return ""
    win = _clip_win(tuple(view)) if view else _clip_win(parse_zoom(zoom_s or ""))
    if win is None:
        return "zoom into a window first"
    z = int(z); rows = []
    for f in MF_FREQS:
        q = iqms_for("mf_stack", z, win, f); r = slice_r12r13("mf_stack", z, win, f)
        rows.append((f, q["edge_width"], q["AES_ratio"], r["r13"]))
    ok = [r for r in rows if np.isfinite(r[1])]
    best = min(ok, key=lambda r: r[1]) if ok else None
    lines = [f"window rows {win[0]}–{win[1]}, cols {win[2]}–{win[3]} at z={z}:   f [Hz]   edge width [px]   AES ×orig   R13 ×orig"
             + (f"     (narrowest edges: {best[0]:+.0f} Hz — check it visually)" if best else "")]
    lines += [f"   {f:+6.0f}      {_fmt(w):>8s}         {_fmt(ae, 3):>6s}      {_fmt(r13, 3):>6s}" for f, w, ae, r13 in rows]
    return "\n".join(lines)


# Synchronised zoom/pan across the three panels (clientside), and the visible window → Store('view').
# The re-entry guard stops the mirrored relayout from re-triggering itself.
app.clientside_callback(
    """
    function(o, a, b) {
        const nu = window.dash_clientside.no_update;
        if (window.__lean_sync) return [nu, nu];
        const ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return [nu, nu];
        const trig = ctx.triggered[0].prop_id || '';
        const sel = function(id){ return document.querySelector('#'+id+' .js-plotly-plot'); };
        const gO = sel('g-orig'), gA = sel('g-a'), gB = sel('g-b');
        let r, src, targets;
        if (trig.indexOf('g-orig') === 0) { r = o; src = gO; targets = [gA, gB]; }
        else if (trig.indexOf('g-a') === 0) { r = a; src = gA; targets = [gO, gB]; }
        else if (trig.indexOf('g-b') === 0) { r = b; src = gB; targets = [gO, gA]; }
        else return [nu, nu];
        if (!r || !window.Plotly || !src) return [nu, nu];
        const keys = Object.keys(r);
        const touchesAxes = keys.some(k => k.indexOf('xaxis') === 0 || k.indexOf('yaxis') === 0);
        if (!touchesAxes) return [nu, nu];
        // read the source panel's actual ranges (after Plotly applied scaleanchor), mirror them
        const lay = src.layout || {};
        const xr = (lay.xaxis && lay.xaxis.range) ? lay.xaxis.range.slice() : null;
        const yr = (lay.yaxis && lay.yaxis.range) ? lay.yaxis.range.slice() : null;
        const auto = (r['xaxis.autorange'] === true) || (r['yaxis.autorange'] === true);
        const u = auto ? {'xaxis.autorange': true, 'yaxis.autorange': 'reversed'} : {'xaxis.range': xr, 'yaxis.range': yr};
        window.__lean_sync = true;
        try { for (const t of targets) { if (t) window.Plotly.relayout(t, u); } }
        finally { setTimeout(function(){ window.__lean_sync = false; }, 120); }
        if (auto || !xr || !yr) return [nu, null];
        const y0 = Math.round(Math.min(yr[0], yr[1])), y1 = Math.round(Math.max(yr[0], yr[1]));
        const x0 = Math.round(Math.min(xr[0], xr[1])), x1 = Math.round(Math.max(xr[0], xr[1]));
        return [nu, [y0, y1, x0, x1]];
    }
    """,
    Output("_sync", "children"), Output("view", "data"),
    Input("g-orig", "relayoutData"), Input("g-a", "relayoutData"), Input("g-b", "relayoutData"),
    prevent_initial_call=True)


if __name__ == "__main__":
    print(f"[lean viewer] {len(METHODS)} volumes, {NZ} slices × {NY}×{NX} → http://127.0.0.1:{ARGS.port}")
    app.run(debug=False, host="127.0.0.1", port=ARGS.port)
