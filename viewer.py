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
ARGS = ap.parse_args()

VOLS: dict[str, np.ndarray] = {"orig": np.load(ARGS.orig, mmap_mode="r")}
LABELS: dict[str, str] = {"orig": "original (uncorrected)"}
_lab_path = os.path.join(ARGS.results, "labels.json")
_labels = json.load(open(_lab_path)) if os.path.exists(_lab_path) else {}
for f in sorted(glob.glob(os.path.join(ARGS.results, "*.npy"))):
    name = os.path.splitext(os.path.basename(f))[0]
    if "_fmap" in name or "_selected_freq" in name or "_patch" in name:
        continue
    VOLS[name] = np.load(f, mmap_mode="r"); LABELS[name] = _labels.get(name, name)
METHODS = list(VOLS.keys())
NZ, NY, NX = VOLS["orig"].shape
_metrics_path = os.path.join(ARGS.results, "metrics.json")
METRICS = json.load(open(_metrics_path)) if os.path.exists(_metrics_path) else {}

_MASK: dict[int, np.ndarray] = {}
_BASE: dict[int, dict] = {}
_SLICE: dict[tuple[str, int], dict] = {}


def mask_for(z):
    if z not in _MASK:
        _MASK[z] = tissue_mask(np.asarray(VOLS["orig"][z]))
    return _MASK[z]


def slice_metrics(method, z):
    key = (method, z)
    if key in _SLICE:
        return _SLICE[key]
    if method in METRICS and "slices" in METRICS[method]:
        s = METRICS[method]["slices"]
        out = {"R12": s["R12"][z], "R13": s["R13"][z], "r12": s["r12"][z], "r13": s["r13"][z]}
    else:
        m = mask_for(z)
        if z not in _BASE:
            _BASE[z] = r12_r13(np.abs(np.asarray(VOLS["orig"][z])), m)
        c = r12_r13(np.abs(np.asarray(VOLS[method][z])), m)
        out = {"R12": c["R12"], "R13": c["R13"], **ratios(_BASE[z], c)}
    _SLICE[key] = out
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
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.H3("Cones off-resonance correction — lean viewer", style={"marginBottom": "2px"}),
    html.Div("Gradient entropy (R12) and normalised gradient energy (R13): gradient CONCENTRATION in the tissue mask, "
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
              Input("url", "search"))
def from_url(search):
    """Shareable views: ?z=42&a=ring_k9_N8&b=autofocus&zoom=217,327,644,754"""
    from urllib.parse import parse_qs
    q = parse_qs((search or "").lstrip("?"))
    a = q.get("a", [METHODS[1] if len(METHODS) > 1 else "orig"])[0]
    b = q.get("b", [METHODS[2] if len(METHODS) > 2 else "orig"])[0]
    z = int(q.get("z", [NZ // 2])[0])
    zoom = q.get("zoom", [""])[0]
    return (a if a in METHODS else METHODS[0], b if b in METHODS else METHODS[0], max(0, min(NZ - 1, z)), zoom)


def parse_zoom(s):
    try:
        v = [int(float(x)) for x in s.replace(";", ",").split(",")]
        return tuple(v) if len(v) == 4 else None
    except Exception:
        return None


@app.callback(Output("g-orig", "figure"), Output("g-a", "figure"), Output("g-b", "figure"),
              Output("t-orig", "children"), Output("t-a", "children"), Output("t-b", "children"),
              Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"),
              Input("show-mask", "value"), Input("zoom-box", "value"))
def update(a, b, z, show_mask, zoom_s):
    z = int(z); show = bool(show_mask); zoom = parse_zoom(zoom_s or "")
    return (make_fig(VOLS["orig"][z], z, show, zoom), make_fig(VOLS[a][z], z, show, zoom), make_fig(VOLS[b][z], z, show, zoom),
            f"original  (z={z})", f"A: {LABELS[a]}", f"B: {LABELS[b]}")


def metric_block(method, z, tag, color):
    s = slice_metrics(method, z)
    v = METRICS.get(method, {}).get("volume")
    rows = [html.Div([html.Span(f"{tag}  ", style={"color": color, "fontWeight": "700"}), html.Span(LABELS[method])]),
            html.Div(f"slice z={z}:   gradient entropy {s['R12']:.3f}  (×{s['r12']:.3f})     "
                     f"normalised gradient energy {s['R13']:.3e}  (×{s['r13']:.3f})")]
    if v:
        rows.append(html.Div(f"full volume:   gradient entropy {v['R12']:.3f}  (×{v['r12']:.3f})     "
                             f"normalised gradient energy {v['R13']:.3e}  (×{v['r13']:.3f})     gain {v['gain']:.3f}",
                             style={"color": "#555"}))
    else:
        rows.append(html.Div("full volume: run compute_metrics.py", style={"color": "#999"}))
    return html.Div(rows, style={"flex": "1", "padding": "6px 10px", "borderLeft": f"3px solid {color}"})


@app.callback(Output("metrics", "children"), Input("a-sel", "value"), Input("b-sel", "value"), Input("z-sel", "value"))
def update_metrics(a, b, z):
    z = int(z)
    blocks = [metric_block(m, z, tag, c) for m, tag, c in ((a, "A", "#1E88E5"), (b, "B", "#E65100")) if m != "orig"]
    if not blocks:
        blocks = [html.Div("select a method")]
    return html.Div(blocks, style={"display": "flex", "gap": "12px"})


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
