"""Run or score the retained phantom and five-volunteer comparisons."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from evaluate import (field_bins, field_support, image_metrics, magnitude,
                      magnitude_gain, mean_field_bins, phantom_psnr)


def load_array(record, directory):
    path = directory / record['file']
    if not path.is_file():
        raise FileNotFoundError(f'Private input is not distributed: {path}')
    if record.get('sha256'):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError(f'Frozen input differs: {path}')
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as bundle:
            array = np.array(bundle[record['key']], copy=True)
    else:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    if record.get('packed_bits'):
        array = np.unpackbits(array, axis=1).astype(bool)
    return array


def load_mask(record, directory, shape):
    array = load_array(record, directory)
    if array.dtype != np.bool_ or array.shape != tuple(shape):
        raise ValueError('Use the exact fixed bool mask with the declared geometry')
    return array


def config(args):
    settings = json.loads(args.inputs.read_text())
    expected = 3 if args.workflow == 'phantom' else 5
    if settings['kind'] != args.workflow or len(settings['cases']) != expected:
        raise ValueError(f'The {args.workflow} comparison requires {expected} fixed cases')
    fixed = dict(shells=16, shell_spacing_ms=.6, updates=3,
                 update_cap_hz=60, final_shells=16)
    if settings.get('correction') != fixed:
        raise ValueError('Use the fixed abstract correction settings; no sweep options are implemented')
    selected = getattr(args, 'case', None)
    if selected is not None:
        if args.action not in ('run', 'check'):
            raise ValueError('--case is supported only for run and check, not cohort scoring')
        cases = [case for case in settings['cases'] if case['id'] == selected]
        if len(cases) != 1:
            raise ValueError(f'--case must match exactly one configured case: {selected}')
        settings = dict(settings, cases=cases)
    return settings


def check_inputs(settings, directory, scoring=False):
    shape = tuple(settings['shape_zxy'])
    for case in settings['cases']:
        for key, dtype in [('volume', np.complex64), ('timing', np.float32), ('reference', np.complex64)]:
            if key == 'reference' and case[key].get('mode') != 'saved_complex':
                raise ValueError('These retained workflows use the saved measured-field CP reference')
            array = load_array(case[key], directory)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"{case['id']}/{key}: expected {dtype.__name__} {shape}")
        if not scoring:
            continue
        if settings['kind'] == 'volunteers':
            for key in ('breast_mask', 'anterior_roi', 'body_mask', 'reference_support'):
                load_mask(case[key], directory, shape)
            for key in ('reference_field_hz', 'reference_magnitude'):
                if load_array(case[key], directory).shape != shape:
                    raise ValueError(f'Wrong field geometry: {key}')
        else:
            for key in ('strict_mask', 'tissue_mask'):
                load_mask(case[key], directory, shape[1:])
            if load_array(case['reference_plane'], directory).shape != shape[1:]:
                raise ValueError('Wrong phantom reference-plane geometry')


def run(settings, directory, output):
    from autofocus import autofocus
    from ring_b0_corr import correct

    check_inputs(settings, directory)
    output.mkdir(parents=True, exist_ok=False)
    for case in settings['cases']:
        target = output / case['id']
        target.mkdir()
        original = load_array(case['volume'], directory)
        timing = load_array(case['timing'], directory)
        image, accumulated, effective, info = correct(
            original, timing, voxel_mm=settings['voxel_mm'], cache_dir=output / 'cache')
        np.save(target / 'isc_complex.npy', image)
        np.save(target / 'accumulated_field_hz.npy', accumulated)
        np.save(target / 'effective_field_hz.npy', effective)
        if settings['kind'] == 'volunteers':
            image, field, af_info = autofocus(original, timing)
            np.save(target / 'autofocus_complex.npy', image)
            np.save(target / 'autofocus_field_hz.npy', field)
            info['autofocus'] = af_info
        # Reuse the validation reference; do not silently reconstruct or register it.
        np.save(target / 'cp_complex.npy', load_array(case['reference'], directory))
        (target / 'settings.json').write_text(json.dumps(info, indent=2) + '\n')
        print(target)


def volunteer_scores(settings, directory, results):
    rows, field_rows, gains = [], {m: [] for m in ('uncorrected', 'autofocus', 'isc')}, {}
    for subject, case in enumerate(settings['cases'], 1):
        original = load_array(case['volume'], directory)
        reference = magnitude(load_array(case['reference'], directory), case['reference_gain'])
        mask = load_mask(case['breast_mask'], directory, original.shape)
        reference_field = load_array(case['reference_field_hz'], directory)
        support = field_support(load_mask(case['anterior_roi'], directory, original.shape),
                                load_mask(case['reference_support'], directory, original.shape),
                                load_array(case['reference_magnitude'], directory),
                                load_mask(case['body_mask'], directory, original.shape))
        gains[case['id']] = {}
        images = {'uncorrected': original,
                  'autofocus': np.load(results / case['id'] / 'autofocus_complex.npy', mmap_mode='r'),
                  'isc': np.load(results / case['id'] / 'isc_complex.npy', mmap_mode='r')}
        fields = {'uncorrected': np.zeros(original.shape, np.float32),
                  'autofocus': np.load(results / case['id'] / 'autofocus_field_hz.npy', mmap_mode='r'),
                  'isc': np.load(results / case['id'] / 'effective_field_hz.npy', mmap_mode='r')}
        for method, image in images.items():
            gain = magnitude_gain(original, image, mask)
            if gain['sampled_slices'] != case['gain_sampling_slices']:
                raise ValueError('The original nonempty-mask gain-sampling slices changed')
            gains[case['id']][method] = gain
            scores = image_metrics(magnitude(image, gain['gain']), reference, mask,
                                   settings['voxel_mm'], case['reference_range'])
            for metric in ('psnr', 'ssim_3d_sigma1p5mm', 'hfen_3d_sigma1p5mm'):
                value = scores[metric]
                if value is None or not np.isfinite(value):
                    raise ValueError('All five finite volunteer scores are required')
                rows.append(dict(subject=subject, case_id=case['id'], method=method,
                                 metric=metric, value=value))
            field_rows[method].append(field_bins(fields[method], reference_field, support))
    means = []
    for method in field_rows:
        for metric in ('psnr', 'ssim_3d_sigma1p5mm', 'hfen_3d_sigma1p5mm'):
            values = [r['value'] for r in rows if (r['method'], r['metric']) == (method, metric)]
            means.append(dict(method=method, metric=metric, equal_volunteer_mean=float(np.mean(values))))
    points = []
    for method, volunteers in field_rows.items():
        points.extend(dict(method=method, **point) for point in mean_field_bins(volunteers))
    return dict(A_case_order=[c['id'] for c in settings['cases']],
                A_method_order=list(field_rows), A_subject_rows=rows,
                A_equal_volunteer_means=means, B_points=points,
                B_statistic='abs(mean_s(mean_x((f_eff-reference)*sign(reference))))', gains=gains)


def phantom_scores(settings, directory, results):
    rows = []
    for case in settings['cases']:
        reference = load_array(case['reference_plane'], directory)
        strict = load_mask(case['strict_mask'], directory, reference.shape)
        tissue = load_mask(case['tissue_mask'], directory, reference.shape)
        original = load_array(case['volume'], directory)
        image = np.load(results / case['id'] / 'isc_complex.npy', mmap_mode='r')
        for method, volume in [('uncorrected', original), ('isc', image)]:
            plane = np.abs(volume[settings['metric_z']]).astype(np.float32)
            scores = phantom_psnr(plane, reference, strict, tissue,
                                  settings['grid_tiles'], settings['reference_range'])
            rows.append(dict(condition=case['id'], method=method, **scores))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'score', 'check'))
    parser.add_argument('workflow', choices=('phantom', 'volunteers'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--case', metavar='ID', help='Run or check one case from the full frozen manifest')
    parser.add_argument('--results', type=Path, help='Saved correction output directory for scoring')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    settings = config(args)
    if args.action == 'check':
        check_inputs(settings, args.data, scoring=True)
        print('Private input geometry and configured file hashes match')
    elif args.action == 'run':
        if args.out is None:
            parser.error('--out is required')
        run(settings, args.data, args.out)
    else:
        if args.out is None or args.results is None:
            parser.error('--results and --out are required for scoring')
        check_inputs(settings, args.data, scoring=True)
        args.out.mkdir(parents=True, exist_ok=False)
        if args.workflow == 'volunteers':
            value = volunteer_scores(settings, args.data, args.results)
            name = 'figure4.json'
        else:
            value, name = phantom_scores(settings, args.data, args.results), 'phantom_scores.json'
        path = args.out / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        print(path)


if __name__ == '__main__':
    main()
