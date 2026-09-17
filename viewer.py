"""Lean slice viewer: original vs two methods, synchronized zoom/pan, per-slice
and full-volume gradient entropy (R12) and normalised gradient energy (R13).

  python viewer.py --orig volume.npy [--results results] [--port 8053]

results/ holds the method volumes written by run_methods.py (<name>.npy, plus
labels.json) and, optionally, metrics.json from compute_metrics.py (used for
the full-volume row and as a cache for the slice values).

Reading the two numbers: both measure how CONCENTRATED the gradient
distribution is inside the tissue mask (ratio vs the original; > 1 = more
concentrated). Deblurring raises them a little — but so do denoising,
smoothing and injected texture, so they are a consistency check next to the
image, not a sharpness score.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import tissue_mask, r12_r13, ratios  # noqa: E402

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
    LABELS["mf_stack"] = f"freq_corr.m at slider frequency (Brian's MF recon, {len(MF_FREQS)} freqs {MF_FREQS[0]:+.0f}..{MF_FREQS[-1]:+.0f} Hz)"
METHODS = list(VOLS.keys())
NZ, NY, NX = VOLS["orig"].shape
_metrics_path = os.path.join(ARGS.results, "metrics.json")
METRICS = json.load(open(_metrics_path)) if os.path.exists(_metrics_path) else {}


def mf_index(f_hz: float) -> int:
    return int(np.argmin(np.abs(np.asarray(MF_FREQS) - f_hz)))


def get_slice(method: str, z: int, f_hz: float = 0.0) -> np.ndarray:
    """|image| slice [rows, cols] for any method, including the MF stack at the slider frequency."""
    if method == "mf_stack":
        return np.abs(np.asarray(MF_VOLS[mf_index(f_hz)][z])).astype(np.float32)
    return np.abs(np.asarray(VOLS[method][z])).astype(np.float32)

_MASK: dict[int, np.ndarray] = {}
_BASE: dict[int, dict] = {}
_SLICE: dict[tuple[str, int], dict] = {}


def mask_for(z):
    if z not in _MASK:
        _MASK[z] = np.asarray(MASK3[z]).astype(bool) if MASK3 is not None else tissue_mask(np.asarray(VOLS["orig"][z]))
    return _MASK[z]


def slice_metrics(method, z, f_hz=0.0):
    key = (method, z, mf_index(f_hz) if method == "mf_stack" else None)
    if key in _SLICE:
        return _SLICE[key]
    if method in METRICS and "slices" in METRICS[method]:
        s = METRICS[method]["slices"]
        out = {"R12": s["R12"][z], "R13": s["R13"][z], "r12": s["r12"][z], "r13": s["r13"][z]}
    else:
        m = mask_for(z)
        if z not in _BASE:
            _BASE[z] = r12_r13(get_slice("orig", z), m)
        c = r12_r13(get_slice(method, z, f_hz), m)
        out = {"R12": c["R12"], "R13": c["R13"], **ratios(_BASE[z], c)}
    _SLICE[key] = out
    return out


def box_metrics(method, z, zoom, f_hz=0.0):
    """Gradient entropy / energy INSIDE the zoom box (the patch being tuned), tissue mask of the box."""
    y0, y1, x0, x1 = zoom
    o = get_slice("orig", z)[y0:y1, x0:x1]
    m = mask_for(z)[y0:y1, x0:x1]
    base = r12_r13(o, m)
    c = r12_r13(get_slice(method, z, f_hz)[y0:y1, x0:x1], m)
    return {"R12": c["R12"], "R13": c["R13"], **ratios(base, c)}


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
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.H3("Cones off-resonance correction — lean viewer", style={"marginBottom": "2px"}),
    html.Div(f"Gradient entropy (R12) and normalised gradient energy (R13): gradient CONCENTRATION in the {'supplied mask (' + os.path.basename(ARGS.mask) + ')' if ARGS.mask else 'Otsu tissue mask'}, "
             "ratio vs original (>1 = more concentrated). Not a sharpness score — judge the images.",
             style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}),
    html.Div([
        html.Div([html.Div("Method A", style=LAB),
                  dcc.Dropdown(id="a-sel", options=[{"label": LABELS[k], "value": k} for k in METHODS],
                               value=METHODS[1] if len(METHODS) > 1 else "orig", clearable=False, style={"width": "380px"})]),
        html.Div([html.Div("Method B", style=LAB),
                  dcc.Dropdown(id="b-sel", options=[{"label": LABELS[k], "value": k} for k in METHODS],
                               value=METHODS[2] if len(METHODS) > 2 else "orig", clearable=False, style={"width": "380px"})]),
        html.Div([html.Div("Zoom box  y0,y1,x0,x1  (blank = full)", style=LAB),
                  dcc.Input(id="zoom-box", type="text", value="", debounce=True, style={"width": "220px"})]),
        html.Div([dcc.Checklist(id="show-mask", options=[{"label": "  tissue-mask outline", "value": "yes"}], value=[])]),
    ], style={"display": "flex", "gap": "14px", "alignItems": "flex-end", "flexWrap": "wrap", "marginBottom": "8px"}),
    html.Div([html.Div(f"MF frequency (Hz) — select 'freq_corr.m at slider frequency' as A or B"
                       + ("" if MF_FREQS else "  [no stack: run run_mf_stack.py]"), style=LAB),
              dcc.Slider(id="f-sel", min=MF_FREQS[0] if MF_FREQS else -150, max=MF_FREQS[-1] if MF_FREQS else 150,
                         value=0, step=None if MF_FREQS else 10,
                         marks={f: f"{f:+.0f}" for f in (MF_FREQS if MF_FREQS else range(-150, 151, 50))},
                         tooltip={"placement": "bottom", "always_visible": True}),
              html.Button("scan f in zoom box", id="scan-btn", n_clicks=0, style={"marginTop": "4px"}),
              html.Div(id="scan-out", style={"fontFamily": "ui-monospace, monospace", "fontSize": "12px", "whiteSpace": "pre"})],
             style={"marginBottom": "8px"}),
    html.Div([html.Div("Slice", style=LAB),
              dcc.Slider(id="z-sel", min=0, max=NZ - 1, value=NZ // 2, step=1,
                         marks={i: str(i) for i in range(0, NZ, max(1, NZ // 8))},
                         tooltip={"placement": "bottom", "always_visible": True})], style={"marginBottom": "8px"}),
    html.Div([
        html.Div([html.H4(id="t-orig", style={"margin": "4px 0"}), dcc.Graph(id="g-orig", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1"}),
        html.Div([html.H4(id="t-a", style={"margin": "4px 0"}), dcc.Graph(id="g-a", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1"}),
        html.Div([html.H4(id="t-b", style={"margin": "4px 0"}), dcc.Graph(id="g-b", config={"scrollZoom": True}, style={"height": "560px"})], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "8px"}),
    html.Div(id="_sync", style={"display": "none"}),
    html.Div(id="metrics", style={"padding": "10px 14px", "border": "1px solid #ddd", "borderRadius": "8px",
                                  "background": "#fafafa", "marginTop": "10px", "fontFamily": "ui-monospace, monospace", "fontSize": "13px"}),
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


def _title(method, f):
    if method == "mf_stack":
        return f"freq_corr.m at {MF_FREQS[mf_index(f)]:+.0f} Hz (Brian's MF recon)"
    return LABELS[method]


@app.callback(Output("g-orig", "figure"), Output("g-a", "figure"), Output("g-b", "figure"),
              Output("t-orig", "children"), Output("t-a", "children"), Output("t-b", "children"),
              Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"),
              Input("show-mask", "value"), Input("zoom-box", "value"), Input("f-sel", "value"))
def update(a, b, z, show_mask, zoom_s, f):
    z = int(z); show = bool(show_mask); zoom = parse_zoom(zoom_s or ""); f = float(f or 0)
    return (make_fig(get_slice("orig", z), z, show, zoom), make_fig(get_slice(a, z, f), z, show, zoom),
            make_fig(get_slice(b, z, f), z, show, zoom),
            f"original  (z={z})", f"A: {_title(a, f)}", f"B: {_title(b, f)}")


def metric_block(method, z, tag, color, zoom, f):
    s = slice_metrics(method, z, f)
    v = METRICS.get(method, {}).get("volume")
    rows = [html.Div([html.Span(f"{tag}  ", style={"color": color, "fontWeight": "700"}), html.Span(_title(method, f))]),
            html.Div(f"slice z={z}:   gradient entropy {s['R12']:.3f}  (×{s['r12']:.3f})     "
                     f"normalised gradient energy {s['R13']:.3e}  (×{s['r13']:.3f})")]
    if zoom is not None:
        bm = box_metrics(method, z, zoom, f)
        rows.append(html.Div(f"zoom box:      gradient entropy {bm['R12']:.3f}  (×{bm['r12']:.3f})     "
                             f"normalised gradient energy {bm['R13']:.3e}  (×{bm['r13']:.3f})", style={"color": "#1F4E79"}))
    if v:
        rows.append(html.Div(f"full volume:   gradient entropy {v['R12']:.3f}  (×{v['r12']:.3f})     "
                             f"normalised gradient energy {v['R13']:.3e}  (×{v['r13']:.3f})     gain {v['gain']:.3f}",
                             style={"color": "#555"}))
    elif method != "mf_stack":
        rows.append(html.Div("full volume: run compute_metrics.py", style={"color": "#999"}))
    return html.Div(rows, style={"flex": "1", "padding": "6px 10px", "borderLeft": f"3px solid {color}"})


@app.callback(Output("metrics", "children"), Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"),
              Input("zoom-box", "value"), Input("f-sel", "value"))
def update_metrics(a, b, z, zoom_s, f):
    z = int(z); zoom = parse_zoom(zoom_s or ""); f = float(f or 0)
    blocks = [metric_block(m, z, tag, c, zoom, f) for m, tag, c in ((a, "A", "#1E88E5"), (b, "B", "#E65100")) if m != "orig"]
    if not blocks:
        blocks = [html.Div("select a method")]
    return html.Div(blocks, style={"display": "flex", "gap": "12px"})


@app.callback(Output("scan-out", "children"), Input("scan-btn", "n_clicks"), Input("z-sel", "value"), Input("zoom-box", "value"),
              prevent_initial_call=True)
def scan_box(n, z, zoom_s):
    """Evaluate every stack frequency inside the zoom box: gradient entropy /
    energy ratios vs the original — an aid for the manual tuning, NOT an
    oracle (gradient concentration also rises with noise/texture)."""
    from dash import ctx
    if not MF_FREQS or ctx.triggered_id != "scan-btn":
        return ""
    zoom = parse_zoom(zoom_s or "")
    if zoom is None:
        return "set a zoom box first"
    z = int(z); rows = []
    for f in MF_FREQS:
        bm = box_metrics("mf_stack", z, zoom, f); rows.append((f, bm["r12"], bm["r13"]))
    best = max(rows, key=lambda r: r[2])
    lines = [f"zoom box {zoom} at z={z}:  f [Hz]   entropy ratio   energy ratio   (highest energy ratio: {best[0]:+.0f} Hz — check it visually)"]
    lines += [f"   {f:+6.0f}      {r12:6.3f}          {r13:6.3f}" for f, r12, r13 in rows]
    return "\n".join(lines)


# Synchronised zoom/pan across the three panels (clientside; re-entry guard
# prevents the mirrored relayout from re-triggering itself).
app.clientside_callback(
    """
    function(o, a, b) {
        if (window.__lean_sync) return window.dash_clientside.no_update;
        const ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        const trig = ctx.triggered[0].prop_id || '';
        const sel = function(id){ return document.querySelector('#'+id+' .js-plotly-plot'); };
        const gO = sel('g-orig'), gA = sel('g-a'), gB = sel('g-b');
        let r, targets;
        if (trig.indexOf('g-orig') === 0) { r = o; targets = [gA, gB]; }
        else if (trig.indexOf('g-a') === 0) { r = a; targets = [gO, gB]; }
        else if (trig.indexOf('g-b') === 0) { r = b; targets = [gO, gA]; }
        else return window.dash_clientside.no_update;
        if (!r || !window.Plotly) return window.dash_clientside.no_update;
        const u = {};
        if (Array.isArray(r['xaxis.range'])) u['xaxis.range'] = r['xaxis.range'].slice();
        else if (r['xaxis.range[0]'] !== undefined) u['xaxis.range'] = [r['xaxis.range[0]'], r['xaxis.range[1]']];
        if (Array.isArray(r['yaxis.range'])) u['yaxis.range'] = r['yaxis.range'].slice();
        else if (r['yaxis.range[0]'] !== undefined) u['yaxis.range'] = [r['yaxis.range[0]'], r['yaxis.range[1]']];
        if (r['xaxis.autorange'] !== undefined) u['xaxis.autorange'] = r['xaxis.autorange'];
        if (r['yaxis.autorange'] !== undefined) u['yaxis.autorange'] = r['yaxis.autorange'];
        if (Object.keys(u).length === 0) return window.dash_clientside.no_update;
        window.__lean_sync = true;
        try { for (const t of targets) { if (t) window.Plotly.relayout(t, u); } }
        finally { setTimeout(function(){ window.__lean_sync = false; }, 90); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("_sync", "children"), Input("g-orig", "relayoutData"), Input("g-a", "relayoutData"), Input("g-b", "relayoutData"),
    prevent_initial_call=True)


if __name__ == "__main__":
    print(f"[lean viewer] {len(METHODS)} volumes, {NZ} slices × {NY}×{NX} → http://127.0.0.1:{ARGS.port}")
    app.run(debug=False, host="127.0.0.1", port=ARGS.port)
