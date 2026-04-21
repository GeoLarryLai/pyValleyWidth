"""Per-transect valley-edge detection via Kneedle on HAND profiles.

For each cross-stream profile, walk outward from the channel on each bank
along the height-above-nearest-drainage (HAND) profile, identify the
dominant outward peak (the ridge with the largest HAND), and locate the
``knee`` of the rising portion leading up to it using a simplified
Kneedle algorithm.  That knee is the valley *edge* on that bank: the
transition from the flat/gently-dipping valley floor to the rising
hillslope.

The two bank edges define a per-transect valley depth
``threshold = min(HAND_left_edge, HAND_right_edge)`` (the lower rim
controls, since water would spill over it).  These thresholds, varying
along the channel, are then painted into a spatially-varying valley
classification grid (``DV``): pixels with ``HAND < threshold_of_nearest_
profile`` are marked valley.  Widths are subsequently measured from that
``DV`` by the standard :func:`swath_width`, so the widths and the DV
polygon are always consistent.
"""

import copy

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    label,
    uniform_filter1d,
)
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _knee_index_kneedle(y, S=1.0):
    """Simplified Kneedle knee finder for a concave-increasing curve.

    Parameters
    ----------
    y : array_like
        1-D array assumed to be (roughly) concave-increasing from ``y[0]``
        at the channel to ``y[-1]`` at the peak.  The corresponding
        ``x`` axis is taken to be ``arange(len(y))``.
    S : float, optional
        Sensitivity gate on the normalised distance-from-diagonal. Higher
        ``S`` requires a more pronounced knee before one is reported.
        Default 1.0.

    Returns
    -------
    int or None
        Index of the knee within ``y``, or ``None`` if the curve is too
        short / too straight / non-finite.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3 or not np.isfinite(y).all():
        return None

    yr = y.max() - y.min()
    xr = float(n - 1)
    if yr <= 0 or xr <= 0:
        return None

    xn = np.arange(n, dtype=np.float64) / xr
    yn = (y - y.min()) / yr

    # For a concave-increasing curve, the knee is where the curve is
    # furthest above the chord (the y=x diagonal after normalisation).
    d = yn - xn
    i = int(np.argmax(d))
    if i == 0 or i == n - 1:
        return None

    # Gate: require the knee to be nontrivially above the diagonal.
    # A perfectly straight line gives d_max == 0; a strong knee gives
    # d_max of order 0.2 - 0.5.  ``S`` linearly scales the gate.
    gate = max(S * 0.01, 1e-6)
    if d[i] < gate:
        return None
    return i


def _max_gradient_edge_index(rising):
    """Index of the steepest outward HAND rise within ``rising``.

    Centred-difference gradient (``np.gradient``) is computed on
    ``rising`` (which spans the channel at index 0 to the dominant peak
    at the last index).  The edge is taken as the location of the
    largest positive gradient.

    Returns
    -------
    int or None
        Index within ``rising`` of the maximum positive derivative, or
        ``None`` if the slice is too short, fully NaN, or has no
        positive gradient (essentially flat profile).
    """
    rising = np.asarray(rising, dtype=np.float64)
    n = rising.size
    if n < 3 or not np.isfinite(rising).any():
        return None

    # ``np.gradient`` on a profile with NaNs propagates them; replace
    # them with -inf so they cannot win the argmax but the array length
    # is preserved.
    if not np.isfinite(rising).all():
        return None

    dh = np.gradient(rising)
    if not np.isfinite(dh).any():
        return None

    i = int(np.argmax(dh))
    if dh[i] <= 0:
        return None
    return i


def _edge_index_for_bank(h, smooth_win=0, prominence=0.0, S=1.0,
                         edge_method='max_gradient'):
    """Locate the valley-edge on one bank of a HAND profile.

    The walk is outward from the channel: ``h[0]`` is at the channel,
    ``h[-1]`` is at the outermost sample of the transect on this bank.

    Strategy (noise-robust):

    1. Optionally smooth ``h`` with a 1-D uniform filter (NaN-safe).
    2. Locate local maxima via :func:`scipy.signal.find_peaks` with a
       minimum ``prominence``.
    3. The *dominant* peak = the one with the greatest HAND value (not
       the first).  Suppresses in-valley noise / terrace bumps.
    4. Locate the valley edge on the rising portion
       ``h[0:peak_idx+1]``:

       * ``edge_method='max_gradient'`` (default) - take the index of
         the largest positive ``dHAND/ds`` (steepest wall).  Falls back
         to the dominant peak if the gradient finder cannot find a
         positive slope (essentially flat profile).
       * ``edge_method='kneedle'`` - apply :func:`_knee_index_kneedle`
         to find the corner of the concave-up rise; fall back to the
         dominant peak if no knee qualifies.

    5. Fallback: if no peak found at all, use ``argmax(h)``.

    Returns
    -------
    edge_idx : int or None
        Index along ``h`` of the detected edge, or ``None`` if no usable
        edge could be found.
    edge_hand : float
        HAND value at the detected edge, or NaN.
    """
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    if n < 3:
        return None, np.nan

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
        return None, np.nan
    if finite.all():
        h_use = h_s
    else:
        first_nan = int(np.argmax(~finite))
        if first_nan < 3:
            return None, np.nan
        h_use = h_s[:first_nan]

    # Find all outward peaks (local maxima).
    prom = float(prominence) if prominence and prominence > 0 else None
    peaks, _props = find_peaks(h_use, prominence=prom)

    if peaks.size > 0:
        dom_peak = int(peaks[int(np.argmax(h_use[peaks]))])
    else:
        # No proper local max - use the absolute argmax as a fallback.
        dom_peak = int(np.nanargmax(h_use))
        if dom_peak == 0:
            return None, np.nan

    rising = h_use[:dom_peak + 1]
    if rising.size < 3:
        edge_idx = dom_peak
    else:
        method = (edge_method or 'max_gradient').lower()
        if method == 'kneedle':
            k = _knee_index_kneedle(rising, S=S)
            edge_idx = dom_peak if k is None else int(k)
        else:  # 'max_gradient' (default) or any unknown -> default
            g = _max_gradient_edge_index(rising)
            edge_idx = dom_peak if g is None else int(g)

    return edge_idx, float(h_use[edge_idx])


def _smooth_thresholds_along_channel(profile_xy, thresholds, radius):
    """NaN-aware moving-mean smoothing of per-transect thresholds.

    For each profile centre, replace its threshold with the mean of the
    finite thresholds whose centres lie within ``radius`` (Euclidean,
    map units).  The query is done with a single KDTree.ball lookup so
    smoothing follows the actual geometry of the channel - profiles on
    different stream segments that happen to be close in space are also
    grouped (this keeps the smoothing well-defined where two segments
    converge).

    Parameters
    ----------
    profile_xy : np.ndarray of shape (N, 2)
        Profile centres (x, y).
    thresholds : np.ndarray of shape (N,)
        Per-transect thresholds (m).  NaN entries are excluded from the
        averaging but their slots still receive a smoothed value if any
        finite neighbour exists in their ball.
    radius : float
        Smoothing radius in map units.  ``radius <= 0`` returns the
        thresholds unchanged.

    Returns
    -------
    np.ndarray of shape (N,)
        Smoothed thresholds.  Slots with no finite neighbours stay NaN.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64).copy()
    profile_xy = np.asarray(profile_xy, dtype=np.float64)
    if radius is None or radius <= 0 or thresholds.size == 0:
        return thresholds

    finite = np.isfinite(thresholds)
    if not finite.any():
        return thresholds

    # Build the KDTree only on finite-threshold profiles, but query for
    # every profile so even (initially) NaN slots get a smoothed value
    # if neighbours exist within the radius.
    xy_v = profile_xy[finite]
    thr_v = thresholds[finite]
    tree = cKDTree(xy_v)
    neighbour_lists = tree.query_ball_point(profile_xy, r=float(radius))

    smoothed = np.full_like(thresholds, np.nan)
    for i, nbrs in enumerate(neighbour_lists):
        if not nbrs:
            continue
        smoothed[i] = float(np.mean(thr_v[nbrs]))
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def per_transect_edge_thresholds(swath_profiles_dz, peak_smoothing_radius=0.0,
                                 peak_min_prominence=0.0,
                                 kneedle_sensitivity=1.0, cellsize=None,
                                 edge_method='max_gradient'):
    """Per-transect bank-edge HAND values and valley-depth threshold.

    Parameters
    ----------
    swath_profiles_dz : list of SwathProfile
        Cross-stream swath profiles whose ``.Z`` is the HAND grid (DZ).
    peak_smoothing_radius : float, optional
        1-D uniform-filter window (m) applied to each per-side HAND
        profile before edge detection. Default 0 (no smoothing).
    peak_min_prominence : float, optional
        Minimum HAND rise (m) required for a sample to count as a local
        maximum.  Filters out in-valley micro-bumps / terraces. Default 0.
    kneedle_sensitivity : float, optional
        Kneedle sensitivity gate on the normalised distance-from-diagonal.
        Used only when ``edge_method='kneedle'``.  Default 1.0.
    cellsize : float, optional
        Grid cell size in map units.  Required for converting
        ``peak_smoothing_radius`` to an integer sample window.  If None,
        inferred from ``disty`` spacing.
    edge_method : {'max_gradient', 'kneedle'}, optional
        Per-bank edge detector.  ``'max_gradient'`` (default) sets the
        edge at the steepest outward HAND rise (``argmax dHAND/ds``)
        within the rising portion to the dominant peak.  ``'kneedle'``
        uses the simplified Kneedle knee finder.  Both fall back to the
        dominant peak itself on degenerate profiles.

    Returns
    -------
    edges : np.ndarray of shape (N_total, 6)
        One row per cross-profile across all swaths, in the same order
        that ``stream2swath`` produces the profiles.  Columns:

        * 0 - profile centre x,
        * 1 - profile centre y,
        * 2 - HAND at left-bank edge (NaN if not found),
        * 3 - HAND at right-bank edge (NaN if not found),
        * 4 - per-transect depth threshold = min(col 2, col 3); NaN if
          neither bank yielded an edge,
        * 5 - saturated flag (1 if at least one bank failed to find
          an edge).
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
            # Positive bank (disty >= 0): channel at index 0 in slice.
            pos_h = profile[c_idx:]
            # Negative bank (disty <= 0): reverse so channel is at index 0
            # and we walk outward.
            neg_h = profile[c_idx::-1]

            _, hand_pos = _edge_index_for_bank(
                pos_h, smooth_win=smooth_win,
                prominence=peak_min_prominence, S=kneedle_sensitivity,
                edge_method=edge_method)
            _, hand_neg = _edge_index_for_bank(
                neg_h, smooth_win=smooth_win,
                prominence=peak_min_prominence, S=kneedle_sensitivity,
                edge_method=edge_method)

            hand_left = hand_neg   # negative-disty bank
            hand_right = hand_pos  # positive-disty bank

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


def _disk_structure(radius_pixels):
    """Return a circular boolean structuring element of the given radius."""
    r = max(int(round(radius_pixels)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def build_dv_from_thresholds(dem, stream, DZ, profile_xy, thresholds,
                             nodata_mask=None, max_dist=None,
                             fill_holes=False, closing_radius=0.0):
    """Build a valley-classification grid from spatially-varying HAND thresholds.

    Each DEM pixel is assigned the threshold of the *nearest* cross-profile
    centre (via a KDTree).  Pixels whose HAND is below that threshold are
    marked valley (1).  Stream pixels are marked channel (2).  Masked
    pixels are forced to hillslope (0).  Optionally, pixels farther from
    any profile than ``max_dist`` are also forced to hillslope, preventing
    the valley fill from leaking into regions that the transect sampling
    never covered.

    Parameters
    ----------
    dem : topotoolbox.GridObject
        Template grid (shape, transform, cellsize).
    stream : topotoolbox.StreamObject
        Stream network whose pixels will be stamped as channel (2).
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
        Maximum planimetric distance (m) from a profile for a pixel to be
        eligible for valley classification.  Pixels beyond this are forced
        to hillslope.  Default None (no distance limit).
    fill_holes : bool, optional
        If True, run :func:`scipy.ndimage.binary_fill_holes` on the
        channel-connected valley mask to close enclosed pockets caused
        by HAND noise inside the valley.  Default False.
    closing_radius : float, optional
        Radius (m) of the disk-shaped structuring element used by
        :func:`scipy.ndimage.binary_closing` to smooth jagged
        boundaries.  ``0`` (default) disables closing.

    Returns
    -------
    DV : topotoolbox.GridObject
        Valley classification grid (0 = hillslope, 1 = valley, 2 = stream).
    """
    nrows, ncols = dem.shape
    dz_arr = np.asarray(DZ.z, dtype=np.float64)
    dv_arr = np.zeros((nrows, ncols), dtype=np.float32)

    srows, scols = stream.node_indices

    valid = np.isfinite(thresholds)
    if valid.any():
        xy_v = np.asarray(profile_xy)[valid]
        thr_v = np.asarray(thresholds)[valid]

        tree = cKDTree(xy_v)

        t = dem.transform
        col_arr = np.arange(ncols)
        row_arr = np.arange(nrows)
        # Pixel centres in map coordinates.
        pix_x = t[0] * col_arr + t[2]
        pix_y = t[4] * row_arr + t[5]
        XX, YY = np.meshgrid(pix_x, pix_y)
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        dists, idx = tree.query(pts, k=1)
        thr_grid = thr_v[idx].reshape(nrows, ncols)
        dist_grid = dists.reshape(nrows, ncols)

        with np.errstate(invalid='ignore'):
            is_valley = np.isfinite(dz_arr) & (dz_arr < thr_grid)

        if max_dist is not None and max_dist > 0:
            is_valley &= (dist_grid <= float(max_dist))

        if nodata_mask is not None:
            is_valley &= ~nodata_mask

        # Restrict the valley to components reachable from the channel by
        # walking through below-threshold HAND cells (same logic as the
        # connected-component step in valleyclass_elev). This discards
        # off-channel low-HAND pockets that the per-profile nearest lookup
        # can otherwise produce near the DEM edges or in disconnected
        # drainages.
        is_valley[srows, scols] = True
        labels, n_labels = label(is_valley,
                                 structure=np.ones((3, 3), dtype=bool))
        keep_mask = np.zeros_like(is_valley)
        if n_labels > 0:
            stream_label_ids = np.unique(labels[srows, scols])
            stream_label_ids = stream_label_ids[stream_label_ids > 0]
            if stream_label_ids.size > 0:
                keep_mask = np.isin(labels, stream_label_ids)

        # Optional morphological cleanup (Kneedle mode in dem2widths).
        if fill_holes and keep_mask.any():
            keep_mask = binary_fill_holes(keep_mask)

        if closing_radius and closing_radius > 0 and keep_mask.any():
            cs = float(getattr(dem, 'cellsize', 1.0)) or 1.0
            radius_pix = max(closing_radius / cs, 1.0)
            keep_mask = binary_closing(
                keep_mask,
                structure=_disk_structure(radius_pix),
            )

        # Re-apply nodata exclusion: closing/fill must not leak into
        # masked regions.
        if nodata_mask is not None:
            keep_mask &= ~nodata_mask

        dv_arr[keep_mask] = 1.0

    # Stream override (always category 2)
    dv_arr[srows, scols] = 2.0

    if nodata_mask is not None:
        dv_arr[nodata_mask] = 0.0

    DV = copy.deepcopy(dem)
    DV.z = dv_arr
    return DV
