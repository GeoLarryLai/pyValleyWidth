"""Map valley-width samples onto a stream network and apply fill + Gaussian smoothing."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .ksn_analysis import fill_nan_stream_segments, smooth_stream_values


def stream_xy_flat_coords(stream_obj):
    """Flatten ``stream_obj.xy()`` map coordinates (same order as fill/smooth)."""
    xs, ys = [], []
    for coord_group in stream_obj.xy():
        for cx, cy in coord_group:
            xs.append(cx)
            ys.append(cy)
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


def _snap_mean_fill_smooth_on_stream(
    allwidths,
    stream_obj,
    max_match_dist,
    window_size,
    width_column,
):
    """Shared snap → mean per vertex → fill segments → smooth along *stream_obj*."""
    stream_coords_flat = stream_xy_flat_coords(stream_obj)
    tree = cKDTree(stream_coords_flat)
    pts = np.column_stack([allwidths[:, 0], allwidths[:, 1]])
    dists, indices = tree.query(pts)

    node_widths = defaultdict(list)
    for i in range(len(dists)):
        if dists[i] < max_match_dist:
            node_widths[indices[i]].append(allwidths[i, width_column])

    vw_net_arr = np.full(len(stream_coords_flat), np.nan, dtype=np.float64)
    for flat_idx, widths in node_widths.items():
        vw_net_arr[flat_idx] = np.mean(widths)

    vw_filled = fill_nan_stream_segments(stream_obj, vw_net_arr)
    vw_net_smooth = smooth_stream_values(stream_obj, vw_filled, window_size=window_size)
    return stream_coords_flat, vw_net_smooth, dists, indices


def valley_width_smoothed_on_stream(
    allwidths,
    stream_obj,
    max_match_dist=150.0,
    window_size=100,
    width_column=2,
):
    """Same pipeline as the smoothed-width map: return flat coords + smoothed vertex field.

    Returns
    -------
    stream_coords_flat : ndarray, (M, 2)
    vw_net_smooth : ndarray, (M,)
        Smoothed width at each ``stream_xy_flat_coords`` vertex.
    dists, indices : ndarray
        KD-tree query from each row of *allwidths* to the flat stream polyline.
    """
    return _snap_mean_fill_smooth_on_stream(
        allwidths, stream_obj, max_match_dist, window_size, width_column
    )


def smooth_valley_width_on_stream_network(
    allwidths,
    stream_obj,
    max_match_dist=150.0,
    window_size=100,
    width_column=2,
):
    """Snap width samples to nearest stream vertices, mean, fill NaNs, smooth along network.

    Parameters
    ----------
    allwidths : ndarray, shape (N, >=3)
        Column *width_column* is interpreted as raw width (m).
    stream_obj : topotoolbox.StreamObject
    max_match_dist : float
        Maximum snap distance (map units, typically m).
    window_size : int
        Passed to ``smooth_stream_values`` (Gaussian sigma scaling).
    width_column : int
        Which column of *allwidths* holds the width to propagate.

    Returns
    -------
    vw_smooth_mapped : ndarray, shape (N,)
        Per-sample smoothed width at the snapped network node (NaN if no snap).
    dists : ndarray
        Query distances from each width point to the nearest stream vertex.
    indices : ndarray
        Nearest stream-flatten index for each width point.
    n_matched : int
        Count of samples with ``dists < max_match_dist``.
    """
    stream_coords_flat, vw_net_smooth, dists, indices = _snap_mean_fill_smooth_on_stream(
        allwidths, stream_obj, max_match_dist, window_size, width_column
    )

    vw_smooth_mapped = np.full(len(allwidths), np.nan, dtype=np.float64)
    for i in range(len(dists)):
        if dists[i] < max_match_dist:
            vw_smooth_mapped[i] = vw_net_smooth[indices[i]]

    n_matched = int(np.sum(dists < max_match_dist))
    return vw_smooth_mapped, dists, indices, n_matched


def xy_flat_nal_walk_order(stream_obj):
    """NAL index at each ``xy()`` visit (same walk as ``fill_nan_stream_segments``)."""
    n_nal = int(stream_obj.stream.size)
    idx_nal = np.arange(n_nal, dtype=np.float32)
    zeros = np.zeros(n_nal, dtype=np.float32)
    order = []
    for seg in stream_obj.xy(data=(idx_nal, zeros)):
        for a, _ in seg:
            order.append(int(a))
    return np.asarray(order, dtype=np.intp)


def nal_from_xyflat_last(stream_obj, values_flat, nal_len=None):
    """Collapse xy-flatten vector to NAL; last visit wins for duplicate nodes."""
    if nal_len is None:
        nal_len = int(stream_obj.stream.size)
    xy_to_nal = xy_flat_nal_walk_order(stream_obj)
    out = np.full(nal_len, np.nan, dtype=float)
    for k, u in enumerate(xy_to_nal):
        out[u] = values_flat[k]
    return out


def xyflat_from_nal_expand(stream_obj, values_nal):
    """Expand NAL vector to xy-flatten length (one value per ``xy()`` visit)."""
    xy_to_nal = xy_flat_nal_walk_order(stream_obj)
    v = np.asarray(values_nal, dtype=float)
    return v[xy_to_nal].copy()


def trunk_valley_width_from_s2_field(
    allwidths,
    s2_stream,
    st_trunk,
    max_match_dist=150.0,
    window_size=100,
    s2_field=None,
    width_column=2,
):
    """Smoothed widths on *s2_stream*, sampled onto *st_trunk* NAL nodes, gap-filled along *st_trunk* only.

    If ``s2_field`` is None, runs ``valley_width_smoothed_on_stream`` on *s2_stream*.
    Otherwise *s2_field* must be ``(s2_coords_flat, s2_vw_smooth_flat)`` from that call
    so map figures and overlays share one field without recomputing.
    """
    if s2_field is None:
        s2_coords, s2_smooth, _, _ = valley_width_smoothed_on_stream(
            allwidths,
            s2_stream,
            max_match_dist=max_match_dist,
            window_size=window_size,
            width_column=width_column,
        )
    else:
        s2_coords, s2_smooth = s2_field

    tree = cKDTree(s2_coords)
    xc, yc = st_trunk.coordinates
    xy = np.column_stack([np.asarray(xc, dtype=float), np.asarray(yc, dtype=float)])
    d, idx = tree.query(xy)
    vw_nal = np.full(xy.shape[0], np.nan, dtype=float)
    mask = d < max_match_dist
    vw_nal[mask] = s2_smooth[idx[mask]]
    vw_flat_st = xyflat_from_nal_expand(st_trunk, vw_nal)
    vw_filled_flat = fill_nan_stream_segments(st_trunk, vw_flat_st)
    return nal_from_xyflat_last(st_trunk, vw_filled_flat, nal_len=xy.shape[0])
