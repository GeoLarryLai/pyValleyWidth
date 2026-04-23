"""End-to-end pipeline: DEM to valley widths.

Translates the MATLAB function ``DEM2widths.m`` from the
ElevationThresholdValleyWidth toolbox.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .valleyclass_elev import valleyclass_elev
from .stream_utils import removeshortstreams, stream2swath
from .swath_width import swath_width
from .peak_valley_width import (
    per_transect_edge_thresholds,
    build_dv_from_thresholds,
)


def dem2widths(dem, streamarea, elevthreshold=None, swath_dx=None, minradius=None,
              swath_width_param='auto', units='pixels', plot=False,
              klargest_conncomps=None, main_trunk_only=False,
              nodata_value='auto', swath_width_quantile=0.99,
              swath_width_safety=1.5,
              max_valley_width=False,
              peak_smoothing_radius=0.0,
              peak_min_prominence=0.0,
              rim_fraction=0.8,
              dv_smooth_radius=None):
    """Full pipeline from DEM to valley-width measurements.

    Parameters
    ----------
    dem : topotoolbox.GridObject
        DEM in projected coordinates (e.g. UTM) with elevation in metres.
    streamarea : int or float
        Drainage-area threshold for stream initiation, in *units*.
    elevthreshold : float, optional
        Vertical distance threshold (m) above stream for valley classification.
        Required when ``max_valley_width=False`` (default); ignored when
        ``max_valley_width=True``.
    swath_dx : float
        Spacing (m) between cross-stream profiles.
    minradius : float
        Radius (m) for the moving-minimum width smoothing, and also the
        minimum stream length for ``removeshortstreams``.
    swath_width_param : float or str, optional
        Total width (m) of the cross-stream profiles. Use ``'auto'``
        (default) to pick a value from the valley classification itself:
        for every stream node, the distance to the nearest hillslope /
        masked pixel is measured via an Euclidean distance transform of
        ``DV``; the chosen total width is
        ``2 * quantile(distances, swath_width_quantile) *
        swath_width_safety``, clipped at half of the DEM's shorter side to
        avoid runaway transects. The auto value is rounded to the nearest
        ``2 * cellsize`` and printed.
    units : str, optional
        Units for *streamarea*: ``'pixels'`` (default), ``'m2'``, etc.
    plot : bool, optional
        If True, show three separate diagnostic figures during valley classification.
    klargest_conncomps : int, optional
        If set (e.g. ``1``), after ``removeshortstreams`` keep only the *k* largest
        connected stream components (``StreamObject.klargestconncomps``), matching
        the main-basin workflow in the Mataian notebook. If ``None`` (default), all
        components on the DEM are retained.
    main_trunk_only : bool, optional
        If True, replace ``S`` with ``S.trunk()`` before valley classification and
        swath extraction so widths (and associated DA / gradient lookups) follow only
        the map-style main stem. Default False keeps the full (filtered) network.
    swath_width_quantile : float, optional
        Quantile (0-1) of per-stream-node distances to wall used when
        ``swath_width_param='auto'``. Default 0.99.
    swath_width_safety : float, optional
        Multiplicative safety factor applied to the quantile-based
        estimate when ``swath_width_param='auto'``. Default 1.5.
    nodata_value : float or str, optional
        Controls NoData masking of the DEM edges.

        * ``'auto'`` (default) - try to detect the sentinel from
          ``dem.nodata`` / ``dem._nodata`` / ``dem.nodataval`` and always mask
          non-finite (NaN / +-Inf) pixels on top. If no sentinel attribute is
          found, only non-finite pixels are masked.
        * a numeric value - mask pixels equal to this sentinel (e.g.
          ``-9999``, ``0``) plus any non-finite pixels.
        * ``None`` - fully disable masking (preserves the original
          pre-fix behavior).

        Masked pixels are forced to hillslope in ``DV`` so empty edges no
        longer register as valley and no longer inflate swath widths.
    max_valley_width : bool, optional
        If True, switch to the variable along-channel elevthreshold
        mode.  For each cross-profile, the HAND profile on each bank is
        scanned outward from the channel; ``scipy.signal.find_peaks``
        (with ``peak_min_prominence``) identifies candidate ridges and
        the *dominant* one (largest HAND) is selected.  The valley rim
        HAND on that bank is declared at ``rim_fraction * H_peak``, and
        the per-transect threshold is the minimum of the two bank rims
        (the lower rim controls).  These per-transect thresholds are
        then interpolated onto the dense stream-node array, smoothed
        along the channel with a rolling median of radius
        ``dv_smooth_radius``, and used to build ``DV`` via the same
        flood-fill logic as :func:`valleyclass_elev` - just with a
        threshold that varies along the channel instead of a single
        scalar.  Widths are measured from that ``DV`` with the same
        :func:`swath_width` method used in the base mode, so widths
        and ``DV`` are always consistent.  ``elevthreshold`` is ignored
        in this mode.  Default False.
    peak_smoothing_radius : float, optional
        Smoothing window (m) applied to each per-side HAND profile
        before rim detection in ``max_valley_width`` mode.  Default 0
        (no smoothing).
    peak_min_prominence : float, optional
        Minimum HAND prominence (m) for a sample to qualify as a ridge
        candidate in ``max_valley_width`` mode.  Filters out minor
        in-valley bumps.  Default 0.
    rim_fraction : float, optional
        Fraction of the dominant peak HAND at which the valley rim is
        declared on each bank in ``max_valley_width=True`` mode.
        Default 0.8.  Smaller values push the rim closer to the valley
        floor (tighter valley); larger values push it toward the ridge
        crest (wider valley).  Ignored when ``max_valley_width=False``.
    dv_smooth_radius : float, optional
        Rolling-median window radius (m) applied along the channel to
        the per-stream-node threshold array in ``max_valley_width=True``
        mode.  Defaults to ``minradius``.  Ignored when
        ``max_valley_width=False``.

    Returns
    -------
    allwidths : np.ndarray
        (N x 7) array. Columns: x, y, raw_width, min_width,
        drainage_area (m^2), stream_gradient (degrees), saturated (0/1 flag
        indicating the transect never hit a hillslope on at least one side).
        Rows with non-finite or non-positive *raw_width* are removed before
        return.
    DV : topotoolbox.GridObject
        Valley classification grid (0 = hillslope, 1 = valley, 2 = stream).
    allvalleydepth : np.ndarray, *only when* ``max_valley_width=True``
        1-D array of length ``allwidths.shape[0]`` giving the per-transect
        valley-depth threshold (m) used to classify each width sample's
        cross-section (lower-rim HAND along the channel).

    The return is a 2-tuple ``(allwidths, DV)`` when ``max_valley_width=False``
    and a 3-tuple ``(allwidths, DV, allvalleydepth)`` when
    ``max_valley_width=True``.
    """
    import topotoolbox as topo

    # --- Validate mode-specific arguments ---
    if max_valley_width:
        if elevthreshold is not None:
            print("max_valley_width=True: ignoring elevthreshold "
                  f"({elevthreshold}); using per-transect peak HAND instead.")
    else:
        if elevthreshold is None:
            raise ValueError(
                "elevthreshold is required when max_valley_width=False."
            )

    # --- Build NoData mask (optional) ---
    nodata_mask = None
    if nodata_value is not None:
        dem_arr = np.asarray(dem.z)

        # Resolve the sentinel value
        if isinstance(nodata_value, str) and nodata_value.lower() == 'auto':
            detected = None
            for attr in ('nodata', '_nodata', 'nodataval'):
                v = getattr(dem, attr, None)
                if v is not None and np.isfinite(v):
                    detected = float(v)
                    break
            if detected is not None:
                print(f"Auto-detected DEM nodata sentinel: {detected}")
            else:
                print(
                    "No finite nodata sentinel attribute found on DEM; "
                    "masking only non-finite (NaN/Inf) pixels."
                )
            sentinel = detected
        else:
            sentinel = float(nodata_value)

        # Always include non-finite pixels in the mask
        nodata_mask = ~np.isfinite(dem_arr)
        if sentinel is not None:
            nodata_mask |= (dem_arr == sentinel)

        if nodata_mask.any():
            print(f"Masking {int(nodata_mask.sum())} NoData pixel(s).")
        else:
            nodata_mask = None

    # --- Flow routing ---
    print("Calculating flow directions")
    FD = topo.FlowObject(dem)

    # --- Flow accumulation & stream network ---
    print("Calculating streams")
    A = FD.flow_accumulation()

    S = topo.StreamObject(FD, threshold=streamarea, units=units)

    # --- Remove short streams ---
    print("Removing short streams")
    S = removeshortstreams(S, minradius)

    if klargest_conncomps is not None:
        k = int(klargest_conncomps)
        if k > 0:
            print(f"Restricting to {k} largest connected stream component(s)")
            S = S.klargestconncomps(k)

    if main_trunk_only:
        print("Restricting stream network to map-style main trunk (StreamObject.trunk)")
        S = S.trunk()

    # --- Valley classification (only needed for the elev-threshold mode) ---
    if not max_valley_width:
        print("Classifying valley")
        DV = valleyclass_elev(dem, S, FD, elevthreshold, plot=plot,
                              nodata_mask=nodata_mask)

    # --- Auto-pick swath_width_param ---
    if isinstance(swath_width_param, str) and swath_width_param.lower() == 'auto':
        if not max_valley_width:
            dv_arr = np.asarray(DV.z)
            not_valley = dv_arr < 1.0  # hillslope / masked pixels act as walls
            dist_px = distance_transform_edt(~not_valley)
            dist_m = dist_px * dem.cellsize
            rows_s, cols_s = S.node_indices
            stream_halfw = dist_m[rows_s, cols_s]
            stream_halfw = stream_halfw[np.isfinite(stream_halfw)]
            if stream_halfw.size == 0:
                fallback = min(dem.shape) * dem.cellsize / 4.0
                print(
                    f"swath_width_param='auto': no valid stream-node distances; "
                    f"falling back to {fallback:.0f} m."
                )
                swath_width_param = float(fallback)
            else:
                q = float(np.quantile(stream_halfw, swath_width_quantile))
                raw = 2.0 * q * swath_width_safety
                # Clip against the DEM's shorter side (full width) so the
                # transect is never longer than half of the DEM.
                cap = 0.5 * min(dem.shape) * dem.cellsize
                if raw > cap:
                    print(
                        f"swath_width_param='auto': estimate {raw:.0f} m exceeds "
                        f"DEM cap {cap:.0f} m; clipping."
                    )
                    raw = cap
                # Round up to an even multiple of cellsize for a symmetric transect
                step = 2.0 * dem.cellsize
                swath_width_param = float(np.ceil(raw / step) * step)
                print(
                    f"swath_width_param='auto': q{swath_width_quantile:.2f} "
                    f"half-width={q:.1f} m -> total width={swath_width_param:.0f} m "
                    f"(safety={swath_width_safety})."
                )
        else:
            # max_valley_width=True: DV not built yet, so use the DEM cap
            # (half of the shorter side) as a generous transect length.
            step = 2.0 * dem.cellsize
            cap = 0.5 * min(dem.shape) * dem.cellsize
            swath_width_param = float(np.ceil(cap / step) * step)
            print(
                f"swath_width_param='auto' (max_valley_width=True): using "
                f"DEM cap = {swath_width_param:.0f} m."
            )

    # --- Swath profiles + width extraction ---
    if max_valley_width:
        print("Computing HAND grid (DZ) for max_valley_width mode")
        DZ = FD.vertdistance2stream(S, dem)
        if nodata_mask is not None:
            dz_arr = np.asarray(DZ.z, dtype=np.float64)
            dz_arr[nodata_mask] = np.nan
            DZ.z = dz_arr

        print("Swath sampling on HAND (for rim detection)")
        swath_profiles_hand = stream2swath(S, DZ, swath_dx, swath_width_param)

        print(
            f"Per-transect rim detection on both banks "
            f"(rim_fraction={rim_fraction})"
        )
        edges = per_transect_edge_thresholds(
            swath_profiles_hand,
            peak_smoothing_radius=peak_smoothing_radius,
            peak_min_prominence=peak_min_prominence,
            rim_fraction=rim_fraction,
            cellsize=dem.cellsize,
        )
        # edges columns: x, y, hand_left, hand_right, threshold, saturated

        smooth_r = (
            float(dv_smooth_radius)
            if dv_smooth_radius is not None
            else float(minradius)
        )
        if smooth_r and smooth_r > 0:
            print(
                f"Rolling-median smoothing of along-channel threshold "
                f"(radius={smooth_r:.0f} m) applied at stream-node resolution"
            )

        print("Building DV from variable along-channel elevthreshold")
        DV = build_dv_from_thresholds(
            dem, S, DZ,
            profile_xy=edges[:, :2],
            thresholds=edges[:, 4],
            nodata_mask=nodata_mask,
            max_dist=swath_width_param / 2.0,
            smooth_radius=smooth_r,
        )

        if plot:
            from .valleyclass_elev import _plot_valley
            _plot_valley(dem, DV, DZ, S)

        print("Swath sampling on DV (for width extraction)")
        swath_profiles = stream2swath(S, DV, swath_dx, swath_width_param)
        raw_widths = swath_width(swath_profiles, minradius)

        # Align per-transect valley-depth thresholds with the raw width rows.
        # Both stream2swath calls used identical S / swath_dx / swath_width_param
        # and stream2swath's profile count depends only on those, so the
        # row ordering matches 1:1.
        if raw_widths.shape[0] == edges.shape[0]:
            per_profile_depth = edges[:, 4].copy()
        else:
            print(
                f"Warning: width rows ({raw_widths.shape[0]}) != edge rows "
                f"({edges.shape[0]}); allvalleydepth will be filled with NaN."
            )
            per_profile_depth = np.full(raw_widths.shape[0], np.nan)
    else:
        print("Swath width extraction")
        swath_profiles = stream2swath(S, DV, swath_dx, swath_width_param)
        raw_widths = swath_width(swath_profiles, minradius)
        per_profile_depth = np.full(raw_widths.shape[0], float(elevthreshold))

    if raw_widths.shape[0] == 0:
        print("Warning: no width measurements produced.")
        return np.empty((0, 7)), DV, np.empty((0,))

    # Split off the saturated flag so DA/gradient slot in before it (final
    # column order: x, y, raw, min, DA, grad, saturated).
    base_widths = raw_widths[:, :4]
    saturated_flag = raw_widths[:, 4]

    # --- Drainage area at width points ---
    print("Extracting drainage area")

    # Build KDTree from actual node coordinates (1:1 with NAL)
    rows, cols = S.node_indices
    t = S.transform
    node_x = t[0] * cols + t[2]
    node_y = t[4] * rows + t[5]
    stream_coords = np.column_stack([node_x, node_y])

    S_A_m = S.ezgetnal(A) * (dem.cellsize ** 2)

    tree = cKDTree(stream_coords)
    _, idx = tree.query(base_widths[:, :2])
    allwidth_DAs = S_A_m[idx]
    allwidths = np.column_stack([base_widths, allwidth_DAs])

    # --- Stream gradient at width points ---
    print("Extracting gradients")
    try:
        SmoZ = S.crs(dem)
        g = S.gradient(SmoZ)
    except Exception:
        g = S.gradient(dem)

    # Convert slope to degrees
    g_deg = np.degrees(np.arctan(np.abs(g)))

    allwidth_grads = g_deg[idx]
    allwidths = np.column_stack([allwidths, allwidth_grads, saturated_flag])

    n_before = allwidths.shape[0]
    raw_w = allwidths[:, 2]
    keep = np.isfinite(raw_w) & (raw_w > 0)
    allwidths = allwidths[keep]
    allvalleydepth = np.asarray(per_profile_depth, dtype=np.float64)[keep]
    n_drop = n_before - allwidths.shape[0]
    if n_drop:
        print(
            f"Discarded {n_drop} width sample(s) with raw_width <= 0 or non-finite."
        )

    n_sat = int(allwidths[:, 6].sum())
    if n_sat:
        print(
            f"Flagged {n_sat} saturated width sample(s) "
            f"(transect never reached a hillslope on at least one side)."
        )

    print(f"Done. {allwidths.shape[0]} width measurements extracted.")
    if max_valley_width:
        return allwidths, DV, allvalleydepth
    return allwidths, DV
