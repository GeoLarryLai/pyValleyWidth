"""Per-transect valley-edge detection via fraction-of-peak HAND.

For each cross-stream profile, each bank's HAND profile is scanned
outward from the channel; the *dominant* outward ridge (tallest HAND
peak meeting ``peak_min_prominence``) is identified, and the valley
*rim* on that bank is defined as the HAND value
``rim_fraction * H_peak``.  The per-transect valley-depth threshold is
the minimum of the two bank rims (the lower rim controls, since water
would spill over it).

These per-transect thresholds are then interpolated onto the *dense*
stream-node array (one threshold per stream cell) and smoothed along
the channel with a NaN-aware rolling median.  Every DEM pixel is
assigned the threshold of its nearest stream node, so the Voronoi
tiles in the threshold raster shrink to a single pixel and the DV
boundary becomes a smooth iso-HAND contour (same character as the
``max_valley_width=False`` flood-fill, but with a threshold that
varies along the channel).  The valley classification grid is built by
thresholding HAND against that per-pixel threshold and keeping only
connected components reachable from the stream - identical logic to
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

def _rim_hand_for_bank(h, smooth_win=0, prominence=0.0, rim_fraction=0.8):
    """Valley-rim HAND on one bank of a HAND profile.

    ``h`` is the outward HAND profile on this bank: ``h[0]`` is at the
    channel, ``h[-1]`` is at the outermost sample.  Strategy:

    1. Optionally smooth ``h`` with a 1-D uniform filter (NaN-safe).
    2. Locate local maxima via :func:`scipy.signal.find_peaks` with a
       minimum ``prominence``.  Fallback to ``argmax(h)`` when no
       qualifying peak exists.
    3. The *dominant* peak = the one with the greatest HAND value
       (suppresses in-valley noise / terrace bumps).
    4. Rim HAND on this bank = ``rim_fraction * H_peak``.

    The rim *location* is intentionally not returned: only the
    threshold value enters the downstream pipeline, and its value is
    far more stable along the channel than any single-point edge-index
    picker (which jumps between terraces).

    Parameters
    ----------
    h : array_like
        Outward HAND profile on this bank (channel at index 0).
    smooth_win : int, optional
        1-D uniform-filter window in samples.  0 disables smoothing.
    prominence : float, optional
        Minimum HAND prominence (m) for a sample to count as a local
        maximum.
    rim_fraction : float, optional
        Fraction of the dominant peak HAND at which the rim is
        declared.  Default 0.8 places the rim near the top of the
        confining wall but below the ridge crest.

    Returns
    -------
    rim_hand : float
        HAND value at the valley rim on this bank, or NaN if no usable
        peak could be found (e.g. the profile is all NaN or flat).
    """
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    if n < 3:
        return np.nan

    if smooth_win and smooth_win >= 2:
        filled = np.where(np.isfinite(h), h, 0.0)
        weights = np.where(np.isfinite(h), 1.0, 0.0)
        num = uniform_filter1d(filled, smooth_win, mode='nearest')
        den = uniform_filter1d(weights, smooth_win, mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            h_s = np.where(den > 0, num / den, np.nan)
        h_s = np.where(np.isfinite(h), h_s, np.nan)
    else:
        h_s = h

    # Truncate at the first NaN encountered walking outward: everything
    # beyond is off-grid / masked and cannot contribute to a ridge.
    finite = np.isfinite(h_s)
    if not finite[0]:
        return np.nan
    if finite.all():
        h_use = h_s
    else:
        first_nan = int(np.argmax(~finite))
        if first_nan < 3:
            return np.nan
        h_use = h_s[:first_nan]

    prom = float(prominence) if prominence and prominence > 0 else None
    peaks, _props = find_peaks(h_use, prominence=prom)

    if peaks.size > 0:
        peak_hand = float(np.max(h_use[peaks]))
    else:
        # No proper local max -> fallback to absolute argmax.
        idx = int(np.nanargmax(h_use))
        if idx == 0:
            return np.nan
        peak_hand = float(h_use[idx])

    if not np.isfinite(peak_hand) or peak_hand <= 0:
        return np.nan

    return float(rim_fraction) * peak_hand


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

def per_transect_edge_thresholds(swath_profiles_dz, peak_smoothing_radius=0.0,
                                 peak_min_prominence=0.0,
                                 rim_fraction=0.8, cellsize=None):
    """Per-transect bank-rim HAND values and valley-depth threshold.

    Parameters
    ----------
    swath_profiles_dz : list of SwathProfile
        Cross-stream swath profiles whose ``.Z`` is the HAND grid (DZ).
    peak_smoothing_radius : float, optional
        1-D uniform-filter window (m) applied to each per-side HAND
        profile before rim detection.  Default 0 (no smoothing).
    peak_min_prominence : float, optional
        Minimum HAND rise (m) required for a sample to count as a local
        maximum.  Filters out in-valley micro-bumps / terraces.  Default 0.
    rim_fraction : float, optional
        Fraction of the dominant peak HAND at which the valley rim is
        declared on each bank.  Default 0.8.
    cellsize : float, optional
        Grid cell size (map units).  Used to convert
        ``peak_smoothing_radius`` to an integer sample window.  If None,
        inferred from ``disty`` spacing.

    Returns
    -------
    edges : np.ndarray of shape (N_total, 6)
        One row per cross-profile across all swaths, in the same order
        :func:`stream2swath` produces the profiles.  Columns:

        * 0 - profile centre x,
        * 1 - profile centre y,
        * 2 - rim HAND on left bank (NaN if not found),
        * 3 - rim HAND on right bank (NaN if not found),
        * 4 - per-transect depth threshold = min(col 2, col 3); NaN if
          neither bank yielded a rim,
        * 5 - saturated flag (1 if at least one bank failed to find a
          rim).
    """
    rows_out = []
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

        smooth_win = 0
        if peak_smoothing_radius and ds > 0:
            smooth_win = int(round(peak_smoothing_radius / ds))
            if smooth_win < 2:
                smooth_win = 0

        c_idx = int(np.argmin(np.abs(disty)))

        for pp in range(n_profiles):
            profile = Z[:, pp]
            pos_h = profile[c_idx:]
            neg_h = profile[c_idx::-1]

            rim_pos = _rim_hand_for_bank(
                pos_h, smooth_win=smooth_win,
                prominence=peak_min_prominence,
                rim_fraction=rim_fraction,
            )
            rim_neg = _rim_hand_for_bank(
                neg_h, smooth_win=smooth_win,
                prominence=peak_min_prominence,
                rim_fraction=rim_fraction,
            )

            hand_left = rim_neg   # negative-disty bank
            hand_right = rim_pos  # positive-disty bank

            left_ok = np.isfinite(hand_left)
            right_ok = np.isfinite(hand_right)
            if left_ok and right_ok:
                threshold = float(min(hand_left, hand_right))
                sat = 0.0
            elif left_ok:
                threshold = float(hand_left)
                sat = 1.0
            elif right_ok:
                threshold = float(hand_right)
                sat = 1.0
            else:
                threshold = np.nan
                sat = 1.0

            rows_out.append([
                float(xy[pp, 0]), float(xy[pp, 1]),
                float(hand_left) if left_ok else np.nan,
                float(hand_right) if right_ok else np.nan,
                threshold, sat,
            ])

    if not rows_out:
        return np.empty((0, 6))
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

        # Step 1: assign each stream node the threshold of its nearest
        # profile centre.  This lifts the sparse per-profile thresholds
        # onto the dense stream-node array (cellsize spacing).
        prof_tree = cKDTree(xy_v)
        _, prof_idx = prof_tree.query(stream_xy, k=1)
        node_thr = thr_v[prof_idx].astype(np.float64, copy=True)

        # Step 2: smooth the per-node threshold along the channel with
        # a NaN-aware rolling median.  The Euclidean KDTree-ball
        # neighbourhood follows the actual stream geometry (handles
        # junctions naturally).
        if smooth_radius and smooth_radius > 0:
            node_thr = _rolling_median_along_channel(
                stream_xy, node_thr, radius=smooth_radius,
            )

        # Step 3: for every DEM pixel, look up the threshold of its
        # nearest stream node.  Because stream nodes are dense, the
        # Voronoi tiles are at pixel scale (no visible scan lines).
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

        # Classify: HAND below the local threshold -> valley candidate.
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

        # Keep only components reachable from the channel by walking
        # through below-threshold HAND cells (identical logic to
        # valleyclass_elev).  This discards off-channel low-HAND
        # pockets.
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
