"""Rebuild the abstract figures from saved scalar results and private image tiles."""
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as E
HERE = Path(__file__).resolve().parent
NS = 'http://www.w3.org/2000/svg'
XL = 'http://www.w3.org/1999/xlink'
E.register_namespace('', NS)
E.register_namespace('xlink', XL)
FG = '#161b20'
COLORS = {'uncorrected': '#7b858e', 'autofocus': '#b57420', 'isc': '#007e75'}

def element(p, tag, **attrs):
    return E.SubElement(p, '{' + NS + '}' + tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})

def text(p, x, y, s, size=20, anchor='start', weight='normal', color=FG, **attrs):
    n = element(p, 'text', x=x, y=y, fill=color, font_family='Arial, Helvetica, sans-serif', font_size=size, text_anchor=anchor, font_weight=weight, **attrs)
    n.text = s
    return n

def path(p, d, color='#7b858e', width=1.6, dash=None, **attrs):
    if dash:
        attrs['stroke_dasharray'] = dash
    return element(p, 'path', d=d, fill='none', stroke=color, stroke_width=width, stroke_linecap='round', **attrs)

def diamond(p, x, y, color, r=8, **attrs):
    return element(p, 'path', d=f'M {x} {y - r} L {x + r} {y} L {x} {y + r} L {x - r} {y} Z', fill='white', stroke=color, stroke_width=2.3, **attrs)

def quantitative_figure(data):
    methods = data['A_method_order']
    cases = data['A_case_order']
    root = E.Element('{' + NS + '}svg', width='1400', height='650', viewBox='0 0 1400 650', version='1.1')
    element(root, 'rect', x=0, y=0, width=1400, height=650, fill='white')
    text(root, 12, 34, 'A', 28, weight='bold')
    text(root, 226, 34, '3D reference agreement', 24, weight='bold')
    path(root, 'M 227 73 L 253 73', '#a4acb3', 1.6)
    element(root, 'circle', cx=240, cy=73, r=4.5, fill='#68737b')
    text(root, 263, 79, 'Volunteer (paired)', 18)
    diamond(root, 459, 73, FG, 6)
    text(root, 478, 79, 'Mean', 18)
    rows = data['A_subject_rows']
    means = data['A_equal_volunteer_means']
    index = {(r['case_id'], r['method'], r['metric']): r['value'] for r in rows}
    ys = [188, 333, 478]
    jitters = [-14, -7, 0, 7, 14]
    specs = [('psnr', 226, 391, 29.1, 35.8, [30, 32, 34], ['30', '32', '34'], 'PSNR ↑'), ('ssim_3d_sigma1p5mm', 423, 588, 0.78, 0.925, [0.8, 0.84, 0.88, 0.92], ['0.80', '0.84', '0.88', '0.92'], '3D SSIM ↑'), ('hfen_3d_sigma1p5mm', 620, 785, 0.08, 0.43, [0.1, 0.2, 0.3, 0.4], ['0.1', '0.2', '0.3', '0.4'], '3D HFEN ↓')]
    for metric, l, r, lo, hi, ticks, labels, title in specs:
        g = element(root, 'g', id='metric_' + metric, data_metric=metric)
        x = lambda v: l + (r - l) * (v - lo) / (hi - lo)
        text(g, (l + r) / 2, 121, title, 21, 'middle', 'bold')
        if metric == 'psnr':
            text(g, (l + r) / 2, 147, 'dB', 18, 'middle')
        for tick, label in zip(ticks, labels):
            path(g, f'M {x(tick)} 157 L {x(tick)} 535', '#e0e3e6', 1.2)
            text(g, x(tick), 568, label, 18, 'middle')
        path(g, f'M {l} 535 L {r} 535', width=1.8)
        for subject, (case, jitter) in enumerate(zip(cases, jitters), 1):
            values = [index[case, m, metric] for m in methods]
            xy = [(x(v), y + jitter) for v, y in zip(values, ys)]
            assert all((lo < v < hi for v in values))
            path(g, ' '.join((('M' if i == 0 else 'L') + f' {xx:.17g} {yy:.17g}' for i, (xx, yy) in enumerate(xy))), '#a4acb3', 1.8, data_case_id=case, data_values=json.dumps(values), data_subject=subject)
            for m, (xx, yy), v in zip(methods, xy, values):
                element(g, 'circle', cx=f'{xx:.17g}', cy=f'{yy:.17g}', r=5.3, fill=COLORS[m], stroke='white', stroke_width=1, data_case_id=case, data_method_id=m, data_metric=metric, data_value=repr(v))
        for m, y in zip(methods, ys):
            v = next((r['equal_volunteer_mean'] for r in means if (r['method'], r['metric']) == (m, metric)))
            assert abs(v - math.fsum((index[c, m, metric] for c in cases)) / 5) < 1e-12
            diamond(g, x(v), y, COLORS[m], data_method_id=m, data_metric=metric, data_mean=repr(v))
    for m, y in zip(methods, ys):
        ss = {'uncorrected': ['Uncorrected'], 'autofocus': ['Autofocus'], 'isc': ['Iterative shell', 'correction', '(3 updates)']}[m]
        for i, s in enumerate(ss):
            text(root, 208, y + (6 if len(ss) == 1 else -21) + i * 27, s, 21, 'end', 'bold' if len(ss) > 1 else 'normal')
    text(root, 842, 34, 'B', 28, weight='bold')
    text(root, 905, 34, 'Effective field map', 24, weight='bold')
    text(root, 905, 63, 'disagreement', 24, weight='bold')
    left, top, side = (936, 115, 420)
    x = lambda v: left + side * v / 200
    y = lambda v: top + side * (1 - v / 200)
    for v in [0, 50, 100, 150, 200]:
        path(root, f'M {left} {y(v)} L {left + side} {y(v)}', '#e0e3e6', 1.2)
        text(root, left - 13, y(v) + 6, str(v), 18, 'end')
        path(root, f'M {x(v)} {top + side} L {x(v)} {top + side + 8}', width=1.7)
        text(root, x(v), 568, str(v), 18, 'middle')
    path(root, f'M {left} {top} L {left} {top + side} L {left + side} {top + side}', width=1.8)
    text(root, left + side / 2, 601, '|Measured reference field| (Hz)', 20, 'middle')
    text(root, 876, top + side / 2, '|Mean directional disagreement| (Hz)', 19, 'middle', transform=f'rotate(-90 876 {top + side / 2})')
    for method in methods:
        pp = [p for p in data['B_points'] if p['method'] == method]
        assert len(pp) == 40
        for p in pp:
            assert p['absolute_mean_signed_disagreement_hz'] == abs(p['signed_mean_hz'])
            assert abs(statistics.mean(p['volunteer_means_hz']) - p['signed_mean_hz']) < 1e-12
        d = ' '.join((('M' if i == 0 else 'L') + f" {x(p['x_hz']):.17g} {y(p['absolute_mean_signed_disagreement_hz']):.17g}" for i, p in enumerate(pp)))
        path(root, d, COLORS[method], 3.4 if method == methods[2] else 2.4, None if method == methods[2] else '3,6', data_method_id=method, data_points=json.dumps(pp))
        if method != methods[0]:
            for p in pp:
                element(root, 'circle', cx=x(p['x_hz']), cy=y(p['absolute_mean_signed_disagreement_hz']), r=2.8, fill=COLORS[method])
    for method, xx, label in zip(methods, [227, 530, 842], ['Uncorrected', 'Autofocus', 'Iterative shell correction (3 updates)']):
        path(root, f'M {xx} 629 L {xx + 29} 629', COLORS[method], 3.2, None if method == methods[2] else '3,5')
        text(root, xx + 39, 635, label, 18)
    return root

def image_figure(number, assets):
    root = E.parse(HERE / f'settings/figure_templates/figure{number}.svg').getroot()
    required = json.loads((HERE / 'settings/figure_images.json').read_text())[str(number)]
    if number == 2:
        with (HERE / 'data/phantom_psnr.csv').open() as stream:
            pairs = list(csv.DictReader(stream))
        labels = [node for node in root.iter() if node.text and node.text.endswith(' dB')]
        if len(labels) != len(pairs):
            raise ValueError('Phantom PSNR labels do not match the saved table')
        for node, pair in zip(labels, pairs):
            node.text = f"{float(pair['uncorrected_psnr_db']):.1f} → {float(pair['isc_psnr_db']):.1f} dB"
    nodes = root.findall('.//{' + NS + '}image')
    if len(nodes) != len(required):
        raise ValueError('Image template does not match its asset list')
    for node, record in zip(nodes, required):
        path = assets / record['file']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'Private saved image differs: {path}')
        key = 'href' if 'href' in node.attrib else '{' + XL + '}href'
        node.set(key, 'data:image/png;base64,' + base64.b64encode(raw).decode())
    return root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=int, nargs='+', choices=(2, 3, 4, 5), default=[4])
    parser.add_argument('--assets', type=Path, help='Private saved image-tile directory for Figures 2, 3 and 5')
    parser.add_argument('--scores', type=Path, default=HERE / 'data/figure4.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if any((n != 4 for n in args.figures)) and args.assets is None:
        parser.error('--assets is required for Figures 2, 3 and 5; MRI images are not distributed')
    import resvg_py
    args.out.mkdir(parents=True, exist_ok=True)
    for number in args.figures:
        if number == 4:
            root = quantitative_figure(json.loads(args.scores.read_text()))
        else:
            root = image_figure(number, args.assets.resolve())
        raw = E.tostring(root, encoding='utf-8', xml_declaration=True)
        target = args.out / f'Figure_{number}'
        target.with_suffix('.svg').write_bytes(raw)
        target.with_suffix('.png').write_bytes(resvg_py.svg_to_bytes(svg_string=raw.decode()))
        print(target.with_suffix('.svg'))
if __name__ == '__main__':
    main()
