"""Per-transect valley-edge detection via width-sensitivity (spillover).

For each cross-stream profile the HAND threshold is swept upward and
the connected valley width at each level is measured.  The derivative
dWidth/dThreshold spikes wherever the valley opens onto a terrace or
plateau.  Two absolute thresholds (in m/m) drive the selection:
``min_dwdt`` discards noise peaks below the floor and ``max_dwdt``
marks divide-scale breakouts; the last qualifying spike before any
breakout defines the per-transect rim threshold.

These per-transect thresholds are then interpolated onto the *dense*
stream-node array (one threshold per stream cell) and smoothed along
the channel with a NaN-aware rolling median.  Every DEM pixel is
assigned the threshold of its nearest stream node, so the Voronoi
tiles in the threshold raster shrink to a single pixel and the DV
boundary becomes a smooth iso-HAND contour (same character as the
``max_valley_width=False`` flood-fill, but with a threshold that
varies along the channel).  The valley classification grid is built by
thresholding HAND against that per-pixel threshold and keeping only
connected components reachable from the stream — identical logic to
:func:`valleyclass_elev.valleyclass_elev`, just with a variable rather
than scalar threshold.
"""

import copy

import numpy as np
from scipy.ndimage import label, uniform_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _clip_profile_to_divides(hand, ci):
    """Clip a HAND cross-section to its local drainage divides.

    Walks outward from the channel index ``ci`` on each side and treats
    the position of the maximum HAND value (``argmax``) as the local
    divide.  All samples beyond a divide are set to NaN so downstream
    width / peak detection automatically caps at the divide.

    Parameters
    ----------
    hand : array_like
        Full cross-section HAND profile (1-D).
    ci : int
        Index of the channel centre in *hand*.

    Returns
    -------
    hand_clipped : np.ndarray
        Copy of *hand* with values past the left/right divides set to
        NaN.
    l_idx, r_idx : int
        Indices of the left and right divides in the original array.
    """
    h = np.asarray(hand, dtype=np.float64).copy()
    right = h[ci:]
    if np.isfinite(right).any():
        r_idx = ci + int(np.nanargmax(right))
        if r_idx + 1 < len(h):
            h[r_idx + 1:] = np.nan
    else:
        r_idx = ci
    left = h[:ci + 1]
    if np.isfinite(left).any():
        l_idx = int(np.nanargmax(left))
        if l_idx > 0:
            h[:l_idx] = np.nan
    else:
        l_idx = ci
    return h, l_idx, r_idx


def _width_sensitivity_threshold(hand, disty, ds,
                                  n_thr=300, dw_smooth=5,
                                  min_dwdt=0.0,
                                  profile_smooth_radius=0.0,
                                  max_dwdt=None,
                                  clip_to_divides=True):
    """Width-sensitivity rim detection on a full cross-section.

    Sweeps the HAND threshold upward, measures the connected valley
    width at each level, and identifies the rim via dWidth/dThreshold
    peaks.  Two absolute thresholds (in m/m, the units of dW/dT)
    control which peaks count:

    * ``min_dwdt`` -- floor: peaks below this value are ignored as
      noise.
    * ``max_dwdt`` -- ceiling: any peak exceeding this is treated as
      a divide-scale breakout; the search stops there and the last
      qualifying peak *before* that breakout is selected.

    Parameters
    ----------
    hand : array_like
        Full cross-section HAND profile (left-to-right, channel near
        the centre).
    disty : array_like
        Perpendicular distances from the stream centre for each sample
        in *hand*.
    ds : float
        Sample spacing (m) along the cross-section.
    n_thr : int
        Number of threshold levels in the sweep.
    dw_smooth : int
        Uniform-filter window (in sweep samples) for smoothing the
        width curve before differentiation.
    min_dwdt : float
        Absolute floor (m/m) on dW/dT for a peak to qualify.  Peaks
        below this value are ignored.  ``0.0`` accepts every detected
        peak.
    profile_smooth_radius : float
        1-D smoothing radius (m) applied to the HAND profile before
        the sweep.  0 disables smoothing.
    max_dwdt : float or None
        Absolute ceiling (m/m) on dW/dT.  Any peak exceeding this
        value is treated as a divide-scale breakout: the search stops
        at that index and the last qualifying peak before it is
        chosen.  ``None`` (default) disables the cap.
    clip_to_divides : bool
        If True (default), clip the cross-section to the local
        drainage divides on each side of the channel (the position
        of max HAND in each half) before running the width sweep.
        Anything beyond a divide is in a different drainage and is
        set to NaN.

    Returns
    -------
    result : dict or None
        ``None`` when detection fails.  Otherwise a dict with keys:

        * ``thr``        – rim HAND threshold (m),
        * ``widths``     – raw width array (one per threshold level),
        * ``widths_s``   – smoothed width array,
        * ``thresholds`` – threshold levels used in the sweep,
        * ``dw``         – dWidth/dThreshold array,
        * ``rim_idx``    – index into *thresholds* of the chosen rim,
        * ``edge_l``     – left valley-edge distance (m),
        * ``edge_r``     – right valley-edge distance (m),
        * ``divide_l``   – left drainage-divide distance (m),
        * ``divide_r``   – right drainage-divide distance (m).
    """
    disty = np.asarray(disty, dtype=np.float64)
    ci = int(np.argmin(np.abs(disty)))
    h = np.asarray(hand, dtype=np.float64)

    sw = int(round(profile_smooth_radius / ds)) if ds > 0 else 0
    if sw >= 2:
        f_ = np.where(np.isfinite(h), h, 0.0)
        w_ = np.where(np.isfinite(h), 1.0, 0.0)
        n_ = uniform_filter1d(f_, sw, mode='nearest')
        d_ = uniform_filter1d(w_, sw, mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            h_s = np.where(d_ > 0, n_ / d_, np.nan)
        h = np.where(np.isfinite(hand), h_s, np.nan)

    if clip_to_divides:
        h, l_div_idx, r_div_idx = _clip_profile_to_divides(h, ci)
        divide_l_val = float(disty[l_div_idx])
        divide_r_val = float(disty[r_div_idx])
    else:
        divide_l_val = float(disty[0])
        divide_r_val = float(disty[-1])

    h_max = np.nanmax(h)
    if not np.isfinite(h_max) or h_max <= 0:
        return None

    thresholds = np.linspace(h_max * 0.005, h_max * 0.95, n_thr)

    # Vectorised edge-finding
    h_right = h[ci:]
    h_left = h[ci::-1]
    d_right = disty[ci:]
    d_left = disty[ci::-1]

    exceed_right = (~np.isfinite(h_right))[None, :] | (h_right[None, :] >= thresholds[:, None])
    exceed_left = (~np.isfinite(h_left))[None, :] | (h_left[None, :] >= thresholds[:, None])

    def _first_true_idx(mask, default_idx):
        any_hit = mask.any(axis=1)
        return np.where(any_hit, np.argmax(mask, axis=1), default_idx)

    r_idx = _first_true_idx(exceed_right, len(d_right) - 1)
    l_idx = _first_true_idx(exceed_left, len(d_left) - 1)

    widths = d_right[r_idx] - d_left[l_idx]

    if dw_smooth >= 3:
        widths_s = uniform_filter1d(widths, dw_smooth, mode='nearest')
    else:
        widths_s = widths.copy()

    dw = np.gradient(widths_s, thresholds)

    dw_max = np.max(dw)
    if dw_max > 1e-6:
        all_pks, _ = find_peaks(dw)

        breakout_idx = None
        if max_dwdt is not None and len(all_pks) > 0:
            over_cap = all_pks[dw[all_pks] > max_dwdt]
            if len(over_cap) > 0:
                breakout_idx = int(over_cap[0])

        candidates = all_pks[all_pks < breakout_idx] if breakout_idx is not None else all_pks

        qualified = candidates[dw[candidates] >= min_dwdt] if len(candidates) > 0 else candidates
        if len(qualified) > 0:
            rim_idx = int(qualified[-1])
        elif len(candidates) > 0:
            rim_idx = int(candidates[int(np.argmax(dw[candidates]))])
        else:
            rim_idx = int(np.argmax(dw))
    else:
        finite_h = h[np.isfinite(h)]
        thr_fb = (float(np.percentile(finite_h, 90))
                  if finite_h.size else h_max * 0.5)
        rim_idx = int(np.argmin(np.abs(thresholds - thr_fb)))
    thr = float(thresholds[rim_idx])

    edge_r_val = float(d_right[r_idx[rim_idx]])
    edge_l_val = float(d_left[l_idx[rim_idx]])

    return dict(thr=thr, widths=widths, widths_s=widths_s,
                thresholds=thresholds, dw=dw, rim_idx=rim_idx,
                edge_l=edge_l_val, edge_r=edge_r_val,
                divide_l=divide_l_val, divide_r=divide_r_val)


def _rolling_median_along_channel(xy, values, radius):
    """NaN-aware rolling-median smoothing of values positioned in 2-D.

    For each point, replace its value with the median of the finite
    values whose positions lie within ``radius`` (Euclidean, map
    units).  A single KDTree.ball lookup handles the neighbourhood so
    geometry is followed naturally even at stream junctions.

    Parameters
    ----------
    xy : np.ndarray of shape (N, 2)
        Point locations (x, y) in map coordinates.
    values : np.ndarray of shape (N,)
        Values to smooth.  NaN entries are excluded from the median
        but their slots still receive a smoothed value if any finite
        neighbour exists in their ball.
    radius : float
        Smoothing radius in map units.  ``radius <= 0`` returns the
        values unchanged.

    Returns
    -------
    np.ndarray of shape (N,)
        Smoothed values.  Slots with no finite neighbours stay NaN.
    """
    values = np.asarray(values, dtype=np.float64).copy()
    xy = np.asarray(xy, dtype=np.float64)
    if radius is None or radius <= 0 or values.size == 0:
        return values

    finite = np.isfinite(values)
    if not finite.any():
        return values

    xy_v = xy[finite]
    val_v = values[finite]
    tree = cKDTree(xy_v)
    neighbour_lists = tree.query_ball_point(xy, r=float(radius))

    smoothed = np.full_like(values, np.nan)
    for i, nbrs in enumerate(neighbour_lists):
        if not nbrs:
            continue
        smoothed[i] = float(np.median(val_v[nbrs]))
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _process_one_profile(args):
    """Worker for a single profile (used by joblib)."""
    (profile, disty, ds, xy_x, xy_y,
     n_thr, dw_smooth, min_dwdt, psr, max_dwdt, clip_div) = args
    ws = _width_sensitivity_threshold(
        profile, disty, ds,
        n_thr=n_thr, dw_smooth=dw_smooth,
        min_dwdt=min_dwdt,
        profile_smooth_radius=psr,
        max_dwdt=max_dwdt,
        clip_to_divides=clip_div,
    )
    if ws is not None:
        thr = ws['thr']
        return [xy_x, xy_y, thr, thr, thr, 0.0]
    return [xy_x, xy_y, np.nan, np.nan, np.nan, 1.0]


def per_transect_edge_thresholds(swath_profiles_dz, cellsize=None,
                                 n_thr=300, dw_smooth=5,
                                 min_dwdt=0.0,
                                 profile_smooth_radius=0.0,
                                 max_dwdt=None,
                                 clip_to_divides=True,
                                 n_jobs=-1):
    """Per-transect spillover-based valley-depth threshold.

    For each cross-section the HAND threshold is swept upward and the
    connected valley width is measured at each level.  The derivative
    dWidth/dThreshold is computed; peaks below ``min_dwdt`` are
    ignored as noise and peaks above ``max_dwdt`` mark a divide-scale
    breakout.  The last qualifying peak before any breakout sets the
    per-transect rim threshold.

    Parameters
    ----------
    swath_profiles_dz : list of SwathProfile
        Cross-stream swath profiles whose ``.Z`` is the HAND grid (DZ).
    cellsize : float, optional
        Grid cell size (map units).  If None, inferred from ``disty``
        spacing.
    n_thr : int
        Number of threshold levels in the sweep.  Default 300.
    dw_smooth : int
        Uniform-filter window (in sweep samples) for smoothing the
        width curve before differentiation.  Default 5.
    min_dwdt : float
        Absolute floor (m/m) on dW/dT for a peak to qualify.  Peaks
        below this are ignored as noise.  Default 0.0 (accept every
        peak).
    profile_smooth_radius : float
        1-D smoothing radius (m) applied to the HAND profile before
        the sweep.  Default 0.0 (disabled).
    max_dwdt : float or None
        Absolute ceiling (m/m) on dW/dT.  Peaks exceeding this value
        are treated as divide-scale breakouts; the search stops there
        and the last qualifying peak before it is chosen.  ``None``
        (default) disables the cap.
    clip_to_divides : bool
        If True (default), each profile is clipped to its local
        drainage divides (max-HAND positions on each side) before the
        width sweep.  Anything beyond a divide is in a different
        drainage and is ignored.
    n_jobs : int
        Number of parallel workers (joblib).  ``-1`` (default) uses
        all available cores.  ``1`` disables parallelism.

    Returns
    -------
    edges : np.ndarray of shape (N_total, 6)
        One row per cross-profile across all swaths, in the same order
        :func:`stream2swath` produces the profiles.  Columns:

        * 0 – profile centre x,
        * 1 – profile centre y,
        * 2 – rim threshold (duplicated for compatibility),
        * 3 – rim threshold (duplicated for compatibility),
        * 4 – per-transect depth threshold; NaN if detection failed,
        * 5 – saturated flag (1 if detection failed).
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    tasks = []
    for swath in swath_profiles_dz:
        Z = swath.Z
        disty = swath.disty
        xy = swath.xy
        n_profiles = xy.shape[0]
        if n_profiles == 0:
            continue

        if (cellsize is None or cellsize <= 0) and disty.size > 1:
            ds = abs(float(np.median(np.diff(disty))))
        else:
            ds = float(cellsize) if cellsize else 0.0

        for pp in range(n_profiles):
            tasks.append((
                Z[:, pp], disty, ds,
                float(xy[pp, 0]), float(xy[pp, 1]),
                n_thr, dw_smooth, min_dwdt, profile_smooth_radius,
                max_dwdt, clip_to_divides,
            ))

    if not tasks:
        return np.empty((0, 6))

    use_parallel = (n_jobs != 1) and len(tasks) > 1
    if use_parallel:
        rows_out = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_one_profile)(t)
            for t in tqdm(tasks, desc='Spillover rim detection',
                          unit='profile')
        )
    else:
        rows_out = [_process_one_profile(t)
                    for t in tqdm(tasks, desc='Spillover rim detection',
                                  unit='profile')]

    return np.asarray(rows_out, dtype=np.float64)


def build_dv_from_thresholds(dem, stream, DZ, profile_xy, thresholds,
                             nodata_mask=None, max_dist=None,
                             smooth_radius=0.0):
    """Build a valley-classification grid from a variable along-channel threshold.

    Per-transect thresholds are first mapped onto the *dense* stream-node
    array (each stream node gets the threshold of its nearest profile
    centre), then smoothed along the channel with a rolling median.  The
    smoothed per-node thresholds define the variable elevthreshold along
    the channel.  For every DEM pixel, the threshold is the one assigned
    to its nearest stream node (KDTree ``k=1`` query); because stream
    nodes are at cellsize spacing, the Voronoi tiles in the threshold
    raster collapse to a single pixel, so the DV boundary behaves like
    a smooth iso-HAND contour - the same visual character as the
    ``valleyclass_elev`` flood-fill, just with a threshold that varies
    along the channel.

    The classification then mirrors :func:`valleyclass_elev` exactly:
    ``is_valley = HAND < threshold``, stream pixels are forced True,
    and only connected components touching a stream pixel are kept.

    Parameters
    ----------
    dem : topotoolbox.GridObject
        Template grid (shape, transform, cellsize).
    stream : topotoolbox.StreamObject
        Stream network whose pixels will be stamped as channel (2) and
        whose nodes host the along-channel variable threshold.
    DZ : topotoolbox.GridObject
        HAND grid (height above nearest drainage).
    profile_xy : np.ndarray of shape (N, 2)
        Cross-profile centres (x, y) aligned with ``thresholds``.
    thresholds : np.ndarray of shape (N,)
        Per-transect valley-depth threshold (m).  NaN entries are
        excluded from the KDTree.
    nodata_mask : np.ndarray of bool, optional
        Pixels to force to hillslope after classification.
    max_dist : float, optional
        Maximum planimetric distance (m) from a stream node for a pixel
        to be eligible for valley classification.  Pixels beyond this
        are forced to hillslope.  Default None (no distance limit).
    smooth_radius : float, optional
        Rolling-median window radius (m), applied along the channel to
        the per-stream-node threshold array.  ``0`` disables smoothing.

    Returns
    -------
    DV : topotoolbox.GridObject
        Valley classification grid (0 = hillslope, 1 = valley, 2 = stream).
    """
    nrows, ncols = dem.shape
    dz_arr = np.asarray(DZ.z, dtype=np.float64)
    dv_arr = np.zeros((nrows, ncols), dtype=np.float32)

    srows, scols = stream.node_indices
    t = dem.transform
    node_x = t[0] * scols + t[2]
    node_y = t[4] * srows + t[5]
    stream_xy = np.column_stack([node_x, node_y])

    valid = np.isfinite(thresholds)
    if valid.any() and stream_xy.shape[0] > 0:
        xy_v = np.asarray(profile_xy)[valid]
        thr_v = np.asarray(thresholds)[valid]

        prof_tree = cKDTree(xy_v)
        _, prof_idx = prof_tree.query(stream_xy, k=1)
        node_thr = thr_v[prof_idx].astype(np.float64, copy=True)

        if smooth_radius and smooth_radius > 0:
            node_thr = _rolling_median_along_channel(
                stream_xy, node_thr, radius=smooth_radius,
            )

        node_tree = cKDTree(stream_xy)
        col_arr = np.arange(ncols)
        row_arr = np.arange(nrows)
        pix_x = t[0] * col_arr + t[2]
        pix_y = t[4] * row_arr + t[5]
        XX, YY = np.meshgrid(pix_x, pix_y)
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        dists, idx = node_tree.query(pts, k=1)
        thr_grid = node_thr[idx].reshape(nrows, ncols)
        dist_grid = dists.reshape(nrows, ncols)

        with np.errstate(invalid='ignore'):
            is_valley = (
                np.isfinite(dz_arr)
                & np.isfinite(thr_grid)
                & (dz_arr < thr_grid)
            )

        if max_dist is not None and max_dist > 0:
            is_valley &= (dist_grid <= float(max_dist))

        if nodata_mask is not None:
            is_valley &= ~nodata_mask

        is_valley[srows, scols] = True
        labels, n_labels = label(is_valley,
                                 structure=np.ones((3, 3), dtype=bool))
        keep_mask = np.zeros_like(is_valley)
        if n_labels > 0:
            stream_label_ids = np.unique(labels[srows, scols])
            stream_label_ids = stream_label_ids[stream_label_ids > 0]
            if stream_label_ids.size > 0:
                keep_mask = np.isin(labels, stream_label_ids)

        if nodata_mask is not None:
            keep_mask &= ~nodata_mask

        dv_arr[keep_mask] = 1.0

    dv_arr[srows, scols] = 2.0

    if nodata_mask is not None:
        dv_arr[nodata_mask] = 0.0

    DV = copy.deepcopy(dem)
    DV.z = dv_arr
    return DV
