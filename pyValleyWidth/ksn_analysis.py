"""Normalized steepness (k_sn) helpers: slope–area fit and stream-ordered smoothing."""

import numpy as np
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def slopearea(stream_obj, dem, acc_grid, plot_fit=True):
    """Bin stream nodes in log(S)–log(A) space, linear fit, return θ and k_s.

    Parameters
    ----------
    stream_obj : topotoolbox.StreamObject
    dem : topotoolbox.GridObject
    acc_grid : topotoolbox.GridObject
        Pixel counts from ``FlowObject.flow_accumulation()`` (same as MATLAB flowacc).
    plot_fit : bool, optional
        If True, show the slope–area diagnostic figure.

    Returns
    -------
    dict
        Keys: ``theta`` (negative concavity index), ``ks``, ``a``, ``g``, ``r_squared``.
    """
    z = stream_obj.ezgetnal(dem)
    a = stream_obj.ezgetnal(acc_grid) * (dem.cellsize ** 2)

    xy_coords = stream_obj.xy()
    xy_data = stream_obj.xy(data=(z, a))

    all_gradients = []
    all_areas = []

    for coord_group, data_group in zip(xy_coords, xy_data):
        if len(coord_group) < 2:
            continue
        coords = np.array(coord_group)
        z_trib, a_trib = zip(*data_group)
        z_trib = np.array(z_trib)
        a_trib = np.array(a_trib)
        diffs = np.diff(coords, axis=0)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs ** 2, axis=1)))])
        grad = np.abs(np.gradient(z_trib, distances))
        grad = np.maximum(grad, 1e-6)
        all_gradients.extend(grad)
        all_areas.extend(a_trib)

    all_gradients = np.array(all_gradients)
    all_areas = np.array(all_areas)

    valid_idx = (
        (all_gradients > 0) & (all_areas > 0)
        & np.isfinite(all_gradients) & np.isfinite(all_areas)
    )
    all_gradients = all_gradients[valid_idx]
    all_areas = all_areas[valid_idx]

    log_area = np.log10(all_areas)
    log_gradient = np.log10(all_gradients)

    n_bins = 50
    bin_edges = np.linspace(log_area.min(), log_area.max(), n_bins)
    bin_indices = np.digitize(log_area, bins=bin_edges)

    binned_log_area = []
    binned_log_gradient = []

    for i in range(1, n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 5:
            binned_log_area.append(np.mean(log_area[mask]))
            binned_log_gradient.append(np.mean(log_gradient[mask]))

    binned_log_area = np.array(binned_log_area)
    binned_log_gradient = np.array(binned_log_gradient)

    slope, intercept, r_value, _p, _std = linregress(
        binned_log_area, binned_log_gradient
    )

    theta = -slope
    ks = 10 ** intercept

    if plot_fit:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(log_area, log_gradient, alpha=0.1, s=1, label='Data', color='gray')
        ax.plot(binned_log_area, binned_log_gradient, 'ro', markersize=6, label='Binned data')
        fit_line = slope * binned_log_area + intercept
        ax.plot(
            binned_log_area, fit_line, 'b-', linewidth=2,
            label=f'θ={theta:.3f}, ks={ks:.2f}, R²={r_value ** 2:.3f}',
        )
        ax.set_xlabel('log10(Area) [m²]', fontsize=12)
        ax.set_ylabel('log10(Gradient) [m/m]', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Slope-Area Relationship', fontsize=14)
        plt.tight_layout()
        plt.show()

    return {
        'theta': -theta,
        'ks': ks,
        'a': 10 ** binned_log_area,
        'g': 10 ** binned_log_gradient,
        'r_squared': r_value ** 2,
    }


def calculate_ksn(stream_obj, dem, acc_grid, theta):
    """Channel steepness index k_sn = S / A^θ along each stream segment."""
    z = stream_obj.ezgetnal(dem)
    a = stream_obj.ezgetnal(acc_grid) * (dem.cellsize ** 2)

    xy_coords = stream_obj.xy()
    xy_data = stream_obj.xy(data=(z, a))

    all_ksn = []

    for coord_group, data_group in zip(xy_coords, xy_data):
        if len(coord_group) < 2:
            all_ksn.extend([np.nan] * len(coord_group))
            continue
        coords = np.array(coord_group)
        z_trib, a_trib = zip(*data_group)
        z_trib = np.array(z_trib)
        a_trib = np.array(a_trib)
        diffs = np.diff(coords, axis=0)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs ** 2, axis=1)))])
        grad = np.abs(np.gradient(z_trib, distances))
        grad = np.maximum(grad, 1e-6)
        ksn_trib = grad / (a_trib ** theta)
        all_ksn.extend(ksn_trib)

    return np.array(all_ksn)


def smooth_stream_values(stream_obj, values, window_size=20):
    """Gaussian smooth *values* independently along each ``stream_obj.xy()`` segment."""
    xy_coords = stream_obj.xy()
    smoothed = []
    idx = 0

    for coord_group in xy_coords:
        if len(coord_group) < 2:
            smoothed.extend([np.nan] * len(coord_group))
            idx += len(coord_group)
            continue
        n_points = len(coord_group)
        vals = values[idx:idx + n_points]
        idx += n_points
        vals_smooth = gaussian_filter1d(vals, sigma=window_size / 5)
        smoothed.extend(vals_smooth)

    return np.array(smoothed)


def fill_nan_stream_segments(stream_obj, values):
    """Linearly interpolate NaNs along each stream segment (xy walk order)."""
    out = np.asarray(values, dtype=float).copy()
    xy_coords = stream_obj.xy()
    idx = 0
    for coord_group in xy_coords:
        n = len(coord_group)
        if n == 0:
            continue
        seg = out[idx: idx + n]
        mask = np.isfinite(seg)
        if not mask.any():
            idx += n
            continue
        x = np.arange(n, dtype=float)
        if mask.sum() == 1:
            seg[:] = seg[mask][0]
        else:
            seg[:] = np.interp(x, x[mask], seg[mask])
        idx += n
    return out
