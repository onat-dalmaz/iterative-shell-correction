"""The image metrics and field-bias reduction used in Figure 4."""
import math

import numpy as np
from scipy import ndimage


def magnitude(image, gain=1.0):
    # Keep the saved float32 magnitude/gain arithmetic before float64 scoring.
    return (np.abs(image).astype(np.float32) * gain).astype(np.float32).astype(np.float64)


def magnitude_gain(original, image, mask):
    slices = [z for z in range(mask.shape[0]) if mask[z].any()][::4]
    if not slices:
        raise ValueError('The fixed breast mask is empty')
    a = np.concatenate([np.abs(original[z]).astype(np.float32)[mask[z]] for z in slices])
    b = np.concatenate([np.abs(image[z]).astype(np.float32)[mask[z]] for z in slices])
    original_mean = float(np.mean(a, dtype=np.float32))
    method_mean = float(np.mean(b, dtype=np.float32))
    if not (original_mean > 0 and method_mean > 0):
        raise ValueError('Positive magnitude means are required')
    return dict(gain=original_mean / method_mean, original_mean=original_mean,
                method_mean=method_mean, sampled_slices=slices)


def _centers(mask, radius):
    # Filter the full image; remove only full-FOV edges, not breast boundaries.
    result = mask.copy()
    for axis, r in enumerate(radius):
        if r:
            low, high = [slice(None)] * 3, [slice(None)] * 3
            low[axis], high[axis] = slice(0, r), slice(-r, None)
            result[tuple(low)] = False
            result[tuple(high)] = False
    return result


def _log(image, spacing):
    sigma = tuple(1.5 / d for d in spacing)
    result = np.zeros(image.shape, np.float64)
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 2
        result += ndimage.gaussian_filter(image, sigma=sigma, order=order,
                                         mode='reflect', truncate=7 / 1.5) / spacing[axis] ** 2
    return result


def image_metrics(actual, reference, mask, spacing_mm, reference_range):
    """Full-volume magnitude PSNR, physical 3D SSIM and normalized 3D HFEN."""
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or mask.dtype != np.bool_ or mask.shape != a.shape:
        raise ValueError('Images and the fixed bool mask must share a 3D shape')
    if a.ndim != 3 or not mask.any() or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError('Finite images and nonempty mask required')
    spacing = tuple(float(d) for d in spacing_mm)
    L = float(reference_range)
    if len(spacing) != 3 or min(spacing) <= 0 or not math.isfinite(L) or L <= 0:
        raise ValueError('Positive millimeter spacings and fixed reference range required')
    residual = a - r
    mse = float(np.sum(residual[mask] ** 2)) / int(mask.sum())
    sigma = tuple(1.5 / d for d in spacing)
    smooth = lambda v: ndimage.gaussian_filter(v, sigma=sigma, mode='reflect', truncate=3.5)
    mu_r, mu_a = smooth(r), smooth(a)
    variance_r = smooth(r * r) - mu_r * mu_r
    variance_a = smooth(a * a) - mu_a * mu_a
    covariance = smooth(a * r) - mu_a * mu_r
    c1, c2 = (.01 * L) ** 2, (.03 * L) ** 2
    ssim_map = ((2 * mu_a * mu_r + c1) * (2 * covariance + c2)) / (
        (mu_a * mu_a + mu_r * mu_r + c1) * (variance_a + variance_r + c2))
    ssim_mask = _centers(mask, tuple(int(3.5 * s + .5) for s in sigma))
    hfen_mask = _centers(mask, tuple(int((7 / 1.5) * s + .5) for s in sigma))
    log_ref, log_error = _log(r, spacing), _log(residual, spacing)
    energy = float(np.sum(log_ref[hfen_mask] ** 2))
    return dict(psnr=10 * math.log10(L * L / mse) if mse > 0 else None,
                perfect_agreement=mse == 0,
                ssim_3d_sigma1p5mm=float(np.mean(ssim_map[ssim_mask])) if ssim_mask.any() else None,
                hfen_3d_sigma1p5mm=math.sqrt(float(np.sum(log_error[hfen_mask] ** 2)) / energy)
                if energy > 0 and hfen_mask.any() else None,
                mask_voxels=int(mask.sum()), ssim_centers=int(ssim_mask.sum()),
                hfen_centers=int(hfen_mask.sum()), mse=mse)


def field_support(anterior_roi, reference_support, reference_magnitude, body_mask):
    """Figure 4B support, distinct from the broad mask for image metrics."""
    shape = anterior_roi.shape
    if not all(a.shape == shape for a in (reference_support, reference_magnitude, body_mask)):
        raise ValueError('Field-support arrays must share a shape')
    if not reference_support.any():
        raise ValueError('Measured-field support is empty')
    threshold = .10 * float(np.percentile(reference_magnitude[reference_support], 99))
    tissue = reference_support & (reference_magnitude > threshold)
    near = np.empty(shape, bool)
    for z in range(shape[0]):
        near[z] = ndimage.distance_transform_edt(ndimage.binary_fill_holes(body_mask[z])) <= 55
    sampled = np.zeros(shape, bool)
    valid = anterior_roi & tissue
    sampled[:, ::2, ::2] = valid[:, ::2, ::2] & near[:, ::2, ::2]
    return sampled


def field_bins(method_field_hz, reference_field_hz, support):
    matched = ndimage.uniform_filter(np.asarray(method_field_hz, np.float32),
                                     size=(4, 3, 3), mode='nearest', origin=0)
    ref = np.asarray(reference_field_hz[support], np.float64)
    error = np.asarray(matched[support], np.float64) - ref
    rows = []
    for low in range(0, 200, 5):
        selected = (np.abs(ref) >= low) & (np.abs(ref) < low + 5)
        rows.append(dict(bin_lo_hz=float(low), bin_hi_hz=float(low + 5),
                         x_hz=low + 2.5, count=int(selected.sum()),
                         signed_mean_hz=float((error[selected] * np.sign(ref[selected])).mean())
                         if selected.any() else None))
    return rows


def mean_field_bins(volunteers):
    """Absolute value after averaging five signed volunteer means, never MAE."""
    if len(volunteers) != 5 or any(len(rows) != 40 for rows in volunteers):
        raise ValueError('All five volunteers and forty 5-Hz bins are required')
    result = []
    for cells in zip(*volunteers):
        low, high = cells[0]['bin_lo_hz'], cells[0]['bin_hi_hz']
        if any((c['bin_lo_hz'], c['bin_hi_hz']) != (low, high) for c in cells):
            raise ValueError('Volunteer bins are not matched')
        counts = [c['count'] for c in cells]
        values = [c['signed_mean_hz'] for c in cells]
        if min(counts) < 50 or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(f'All five volunteers need >=50 voxels in [{low}, {high}) Hz')
        signed = float(np.mean(np.array(values, np.float64)))
        result.append(dict(bin_lo_hz=low, bin_hi_hz=high, x_hz=cells[0]['x_hz'],
                           counts=counts, volunteer_means_hz=values, signed_mean_hz=signed,
                           absolute_mean_signed_disagreement_hz=abs(signed)))
    return result


def phantom_psnr(candidate, reference, strict_mask, tissue_mask, tiles, reference_range):
    """Mean of the fixed finite grid-tile PSNR values used in Figure 2."""
    yy, xx = np.ogrid[-3:4, -3:4]
    mask = strict_mask & ndimage.binary_dilation(tissue_mask, structure=xx * xx + yy * yy <= 9)
    c, r = np.asarray(candidate, np.float64) / reference_range, np.asarray(reference, np.float64) / reference_range
    rows = []
    for tile in tiles:
        index = (slice(*tile['x']), slice(*tile['y']))
        selected = mask[index]
        mse = float(np.mean((c[index][selected] - r[index][selected]) ** 2)) if selected.any() else None
        value = -10 * math.log10(mse) if mse is not None and mse > 0 else None
        rows.append(dict(tile=tile['id'], count=int(selected.sum()), psnr_db=value,
                         perfect_agreement=mse == 0))
    finite = [r['psnr_db'] for r in rows if r['psnr_db'] is not None and math.isfinite(r['psnr_db'])]
    return dict(psnr_db=float(np.mean(finite)) if finite else None,
                finite_tiles=len(finite), per_tile=rows)
