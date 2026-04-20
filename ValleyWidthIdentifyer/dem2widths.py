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


def dem2widths(dem, streamarea, elevthreshold, swath_dx, minradius,
              swath_width_param='auto', units='pixels', plot=False,
              klargest_conncomps=None, main_trunk_only=False,
              nodata_value='auto', swath_width_quantile=0.99,
              swath_width_safety=1.5):
    """Full pipeline from DEM to valley-width measurements.

    Parameters
    ----------
    dem : topotoolbox.GridObject
        DEM in projected coordinates (e.g. UTM) with elevation in metres.
    streamarea : int or float
        Drainage-area threshold for stream initiation, in *units*.
    elevthreshold : float
        Vertical distance threshold (m) above stream for valley classification.
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
    """
    import topotoolbox as topo

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

    # --- Valley classification ---
    print("Classifying valley")
    DV = valleyclass_elev(dem, S, FD, elevthreshold, plot=plot,
                          nodata_mask=nodata_mask)

    # --- Auto-pick swath_width_param from DV if requested ---
    if isinstance(swath_width_param, str) and swath_width_param.lower() == 'auto':
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

    # --- Swath profiles ---
    print("Swath width extraction")
    swath_profiles = stream2swath(S, DV, swath_dx, swath_width_param)
    raw_widths = swath_width(swath_profiles, minradius)

    if raw_widths.shape[0] == 0:
        print("Warning: no width measurements produced.")
        return np.empty((0, 7)), DV

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
    return allwidths, DV
