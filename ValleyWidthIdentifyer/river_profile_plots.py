"""Map-view stream plots, longitudinal and χ–Z profiles, excess topography figures."""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .ksn_analysis import calculate_ksn, fill_nan_stream_segments, smooth_stream_values
from .valley_width_stream import trunk_valley_width_from_s2_field, valley_width_smoothed_on_stream


def _network_longprofile_dist_km_z(S_all, dem):
    """Outlet-distance (km) and elevation at each stream node (Mataian all-stream scatter).

    ``S.distance()`` is already a node-attribute list of cumulative upstream distance
    from the outlet (in metres), so we use it directly — no edge integration needed.
    """
    s_z = S_all.ezgetnal(dem)
    x_km = np.asarray(S_all.distance(), dtype=np.float64) / 1000.0
    z_m = np.asarray(s_z, dtype=np.float64)
    n = min(len(x_km), len(z_m))
    return x_km[:n], z_m[:n]


def longest_tributary_subgraph(stream_obj):
    """Return a ``StreamObject`` containing only the longest DFS segment (main channel proxy)."""
    n = int(stream_obj.stream.size)
    ix = np.arange(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    parts = stream_obj.xy(data=(ix, z))
    best = max(parts, key=len)
    nodes = {int(round(a)) for a, _ in best}
    keep = np.zeros(n, dtype=bool)
    for u in nodes:
        if 0 <= u < n:
            keep[u] = True
    return stream_obj.subgraph(keep)


def plot_stream_ksn(
    stream_obj,
    dem,
    ksn_values,
    title='Normalized Channel Steepness (ksn)',
    vmin=0,
    vmax=None,
    figsize=(15, 10),
    savefig=None,
):
    """Hillshade + stream nodes colored by *ksn_values* (same walk order as ``calculate_ksn``)."""
    fig, ax = plt.subplots(figsize=figsize)
    dem.plot_hs(ax=ax, cmap='gray', norm=mcolors.Normalize(vmin=np.nanmin(np.asarray(dem)), vmax=np.nanmax(np.asarray(dem))))

    xy_coords = stream_obj.xy()
    idx = 0
    scatter = None
    # If vmax is not provided, use the data maximum (after vmin clipping).
    ksn_vmax = vmax
    if vmax is None:
        ksn_vmax = np.nanmax(ksn_values)
    for coord_group in xy_coords:
        if len(coord_group) < 2:
            idx += len(coord_group)
            continue
        n_points = len(coord_group)
        ksn_vals = np.clip(ksn_values[idx:idx + n_points], vmin, ksn_vmax)
        idx += n_points
        x_coords, y_coords = zip(*coord_group)
        scatter = ax.scatter(
            x_coords, y_coords, c=ksn_vals,
            cmap='viridis', s=5, vmin=vmin, vmax=ksn_vmax,
            edgecolors='none', alpha=0.8,
        )

    if scatter is not None:
        cbar = plt.colorbar(scatter, label='ksn', ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_longitudinal_trunk_colored_ksn(
    S_all,
    st_trunk,
    dem,
    ksn_st_smoothed,
    vmin=0,
    vmax=None,
    theta=None,
    figsize=(10, 8),
    savefig=None,
):
    """Gray branches as in χ–Z: point per branch colored node, trunk colored by k_sn by distance."""

    # Per-node cumulative upstream distance (m) from ``S.distance()`` — already the
    # correct node-attribute list, no trapz integration needed.
    s2_z = S_all.ezgetnal(dem)
    dist_s2_m = np.asarray(S_all.distance(), dtype=np.float64)
    all_dist_km = dist_s2_m / 1000.0
    all_z = np.asarray(s2_z, dtype=np.float64)

    # Branches: panel-style as in plot_chi_z_trunk_colored_gray_network
    # Plot all branches as faint gray points (excluding trunk)
    st_trunk_nodes = set(st_trunk.nodes) if hasattr(st_trunk, 'nodes') else set()
    if not st_trunk_nodes:
        # If not present, try to recover from S_all.trunk()
        st_trunk_nodes = set(getattr(S_all.trunk(), 'nodes', []))
    # Compose per-branch node indices via xy
    s2_z = S_all.ezgetnal(dem)
    s2_dist = all_dist_km
    groups = S_all.xy(data=(s2_dist, s2_z))

    background_dist = []
    background_z = []
    for seg in groups:
        if not seg:
            continue
        # Exclude trunk if possible (by node index if available)
        this_indices = [i for i, _ in enumerate(seg)]
        # Attempt to identify trunk polylines via set membership on nodes (if available)
        # Fallback: Try to skip trunk by comparing segment length to trunk length
        # (If longest, assume trunk)
        # Let's filter nodes if .nodes are available.
        is_trunk_seg = False
        if hasattr(S_all, "nodes") and hasattr(st_trunk, "nodes"):
            seg_node_indices = [S_all.nodes[i] for i in range(len(S_all.nodes)) if i < len(seg)]
            if set(seg_node_indices).intersection(st_trunk_nodes):
                # This segment contains the trunk nodes; skip
                is_trunk_seg = True
        if not is_trunk_seg:
            # Get d,z by tuple expansion
            dvals, zvals = zip(*seg)
            background_dist.extend(dvals)
            background_z.extend(zvals)

    background_dist = np.asarray(background_dist)
    background_z = np.asarray(background_z)

    fig, ax = plt.subplots(figsize=figsize)
    # Use similar gray style to plot_chi_z_trunk_colored_gray_network (s=1, alpha=0.3)
    ax.scatter(
        background_dist,
        background_z,
        c='gray',
        s=1,
        alpha=0.3,
        linewidths=0,
        edgecolors='none',
        zorder=1,
    )

    # Trunk: per-node upstream distance from outlet (m → km)
    st_z = st_trunk.ezgetnal(dem)
    trunk_dist_km = np.asarray(st_trunk.distance(), dtype=np.float64) / 1000.0

    min_len = min(len(trunk_dist_km), len(st_z), len(ksn_st_smoothed))
    trunk_dist_km = trunk_dist_km[:min_len]
    st_z = st_z[:min_len]
    ksn_plot = ksn_st_smoothed[:min_len]

    # Automatically set vmax if not provided
    ksn_vmax = vmax
    if vmax is None:
        ksn_vmax = np.nanmax(ksn_plot)

    sc = ax.scatter(
        trunk_dist_km, st_z, c=ksn_plot, cmap='viridis',
        s=8, vmin=vmin, vmax=ksn_vmax, alpha=1, linewidths=0, edgecolors='none', zorder=5,
    )
    if theta is not None:
        cbar_lbl = rf'$k_{{sn}}$ ($\theta$ = {abs(float(theta)):.2f})'
    else:
        cbar_lbl = r'$k_{sn}$'
    plt.colorbar(sc, ax=ax, label=cbar_lbl)
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title('Longitudinal profile: trunk colored by $k_{sn}$, other channels gray', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_longitudinal_trunk_ksn_valley_width_twin(
    s2,
    st,
    dem,
    dem_imposed,
    acc,
    allwidths,
    theta=-0.45,
    ksn_window=200,
    valleywidth_window=100,
    max_match_dist=150.0,
    vw_outline_sigma_pts=4.0,
    ksn_vmin=0,
    ksn_vmax=None,
    ksn_clim=None,
    valleywidth_ylim=None,
    figsize=(16, 6),
    savefig=None,
    allvalleydepth=None,
    valleydepth_window=None,
    ma_window_m=500.0,
):
    """Long profile (distance-Z): gray s2 scatter, trunk colored by k_sn, twin = valley width.

    Three-layer valley-width / valley-top overlay:

    1. **Stream-network smoothed** values at each trunk NAL vertex (width via
       ``valley_width_smoothed_on_stream`` + ``trunk_valley_width_from_s2_field`` with
       window ``valleywidth_window``; depth via the same pipeline with
       ``valleydepth_window``, falling back to ``valleywidth_window``) are plotted as
       faint dots (``alpha=0.2``).
    2. A **distance-based moving-average curve** over those smoothed dots, with window
       ``ma_window_m`` meters, shared between width and valley-top.
    3. Shading (``alpha=0.25`` for width, ``alpha=0.3`` for valley top) is drawn under /
       between the MA curve(s) only.

    Axes are in meters by default. When the max magnitude of distance or width exceeds
    10 000, values are divided by 1000 and the axis label becomes ``(x1000 m)``.

    Note: ``vw_outline_sigma_pts`` is retained for API stability but is no longer used
    (the moving average replaces the old outline smoothing).
    """
    st_z = np.asarray(st.ezgetnal(dem), dtype=float)
    trunk_dist_m = np.asarray(st.distance(), dtype=np.float64)

    ksn_st = calculate_ksn(st, dem_imposed, acc, theta)
    ksn_st_smoothed = smooth_stream_values(st, ksn_st, window_size=ksn_window)

    s2_coords_flat, s2_vw_smooth_flat, dists_aw, _ = valley_width_smoothed_on_stream(
        allwidths, s2, max_match_dist=max_match_dist, window_size=valleywidth_window
    )
    vw_trunk_smooth_nal = trunk_valley_width_from_s2_field(
        allwidths,
        s2,
        st,
        max_match_dist=max_match_dist,
        window_size=valleywidth_window,
        s2_field=(s2_coords_flat, s2_vw_smooth_flat),
    )

    # Optional valley-depth field projected onto the trunk via the same s2 pipeline.
    depth_trunk_smooth_nal = None
    if allvalleydepth is not None:
        depth_arr = np.asarray(allvalleydepth, dtype=float).reshape(-1)
        depth_xyz = np.column_stack(
            [allwidths[:, 0], allwidths[:, 1], depth_arr]
        )
        d_window = (
            int(valleydepth_window)
            if valleydepth_window is not None
            else int(valleywidth_window)
        )
        s2_d_coords, s2_d_smooth, _, _ = valley_width_smoothed_on_stream(
            depth_xyz, s2, max_match_dist=max_match_dist, window_size=d_window
        )
        depth_trunk_smooth_nal = trunk_valley_width_from_s2_field(
            depth_xyz,
            s2,
            st,
            max_match_dist=max_match_dist,
            window_size=d_window,
            s2_field=(s2_d_coords, s2_d_smooth),
        )

    min_len = min(
        len(trunk_dist_m), len(st_z), len(ksn_st_smoothed), len(vw_trunk_smooth_nal)
    )
    if depth_trunk_smooth_nal is not None:
        min_len = min(min_len, len(depth_trunk_smooth_nal))
    trunk_dist_m = np.asarray(trunk_dist_m[:min_len], dtype=np.float64)
    st_z = st_z[:min_len]
    ksn_st_smoothed = ksn_st_smoothed[:min_len]
    vw_trunk_smooth_nal = np.asarray(vw_trunk_smooth_nal[:min_len], dtype=np.float64)
    if depth_trunk_smooth_nal is not None:
        depth_trunk_smooth_nal = np.asarray(
            depth_trunk_smooth_nal[:min_len], dtype=np.float64
        )

    # Distance-based NaN-aware moving average over sorted (d, v).
    def _moving_avg_distance(d_sorted_m, v_sorted, window_m):
        v = np.asarray(v_sorted, dtype=float)
        if window_m is None or window_m <= 0 or v.size == 0:
            return v.copy()
        half = float(window_m) / 2.0
        lo = np.searchsorted(d_sorted_m, d_sorted_m - half, side='left')
        hi = np.searchsorted(d_sorted_m, d_sorted_m + half, side='right')
        flag = np.isfinite(v).astype(float)
        filled = np.where(np.isfinite(v), v, 0.0)
        csum = np.concatenate([[0.0], np.cumsum(filled)])
        cflag = np.concatenate([[0.0], np.cumsum(flag)])
        num = csum[hi] - csum[lo]
        den = cflag[hi] - cflag[lo]
        out = np.full_like(v, np.nan)
        nz = den > 0
        out[nz] = num[nz] / den[nz]
        return out

    order = np.argsort(trunk_dist_m)
    xd_sorted_m = trunk_dist_m[order]
    z_ch_sorted = st_z[order]
    vw_smooth_sorted = vw_trunk_smooth_nal[order]
    vw_ma_sorted = _moving_avg_distance(xd_sorted_m, vw_smooth_sorted, ma_window_m)

    valley_top_sorted = None
    valley_top_ma_sorted = None
    if depth_trunk_smooth_nal is not None:
        depth_smooth_sorted = depth_trunk_smooth_nal[order]
        valley_top_sorted = z_ch_sorted + depth_smooth_sorted
        valley_top_ma_sorted = _moving_avg_distance(
            xd_sorted_m, valley_top_sorted, ma_window_m
        )

    if ksn_clim is not None:
        _ksn_vmin, _ksn_vmax = ksn_clim
    else:
        _ksn_vmin = ksn_vmin
        _ksn_vmax = ksn_vmax
    if _ksn_vmax is None:
        _ksn_vmax = np.nanmax(ksn_st_smoothed)
    if _ksn_vmin is None:
        _ksn_vmin = 0

    mask_close = dists_aw < max_match_dist
    print(
        f'Valley width points matched to s2 network: {mask_close.sum()} / {len(dists_aw)}'
    )

    theta_lbl = f'{abs(theta):.2f}'

    # Auto-scale axis units: (m) by default, switch to (x1000 m) when max magnitude > 1e4.
    def _axis_scale_label(values, base_label):
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        vmax = float(np.max(np.abs(arr[finite]))) if finite.any() else 0.0
        if vmax > 10000.0:
            return 1.0 / 1000.0, f'{base_label} (x1000 m)'
        return 1.0, f'{base_label} (m)'

    scale_x, xlabel = _axis_scale_label(trunk_dist_m, 'Distance')
    scale_z, zlabel = _axis_scale_label(st_z, 'Elevation')
    scale_w, wlabel = _axis_scale_label(vw_trunk_smooth_nal, 'Valley Width')

    fig, ax_main = plt.subplots(figsize=figsize)

    # Per-node upstream distance on the s2 network (metres) for the gray background.
    all_dist_m = np.asarray(s2.distance(), dtype=np.float64)
    s2_z = s2.ezgetnal(dem)
    all_z = np.asarray(s2_z, dtype=np.float64)  # noqa: F841 -- left for clarity

    st_trunk_nodes = set(st.nodes) if hasattr(st, 'nodes') else set()
    if not st_trunk_nodes:
        st_trunk_nodes = set(getattr(s2.trunk(), 'nodes', []))
    groups = s2.xy(data=(all_dist_m, np.asarray(s2_z, dtype=np.float64)))
    background_dist = []
    background_z = []
    for seg in groups:
        if not seg:
            continue
        is_trunk_seg = False
        if hasattr(s2, 'nodes') and hasattr(st, 'nodes'):
            seg_node_indices = [s2.nodes[i] for i in range(len(s2.nodes)) if i < len(seg)]
            if set(seg_node_indices).intersection(st_trunk_nodes):
                is_trunk_seg = True
        if not is_trunk_seg:
            dvals, zvals = zip(*seg)
            background_dist.extend(dvals)
            background_z.extend(zvals)
    background_dist = np.asarray(background_dist)
    background_z = np.asarray(background_z)

    ax_main.scatter(
        background_dist * scale_x,
        background_z * scale_z,
        c='gray',
        s=1,
        alpha=0.3,
        linewidths=0,
        edgecolors='none',
        zorder=1,
    )

    # Valley-top envelope: smoothed-on-s2 dots + distance-based MA curve + shading.
    if valley_top_ma_sorted is not None:
        valid_top_dots = np.isfinite(valley_top_sorted)
        if valid_top_dots.any():
            ax_main.scatter(
                xd_sorted_m[valid_top_dots] * scale_x,
                valley_top_sorted[valid_top_dots] * scale_z,
                c='red',
                s=6,
                alpha=0.2,
                linewidths=0,
                edgecolors='none',
                zorder=3,
                label='Valley top (smoothed on s2)',
            )
        valid_top_ma = np.isfinite(z_ch_sorted) & np.isfinite(valley_top_ma_sorted)
        if valid_top_ma.any():
            ax_main.fill_between(
                xd_sorted_m[valid_top_ma] * scale_x,
                z_ch_sorted[valid_top_ma] * scale_z,
                valley_top_ma_sorted[valid_top_ma] * scale_z,
                color='red',
                alpha=0.3,
                linewidth=0,
                zorder=2,
                label='Valley depth envelope',
            )
            ax_main.plot(
                xd_sorted_m[valid_top_ma] * scale_x,
                valley_top_ma_sorted[valid_top_ma] * scale_z,
                color='red',
                alpha=0.6,
                linewidth=1.2,
                zorder=4,
                label=f'Valley top (MA, {ma_window_m:g} m)',
            )

    sc1 = ax_main.scatter(
        trunk_dist_m * scale_x,
        st_z * scale_z,
        c=ksn_st_smoothed,
        cmap='viridis',
        s=8,
        vmin=_ksn_vmin,
        vmax=_ksn_vmax,
        linewidths=0,
        edgecolors='none',
        zorder=5,
        alpha=0.9,
    )
    cbar1 = plt.colorbar(sc1, ax=ax_main, pad=0.08)
    cbar1.set_label(rf'$k_{{sn}}$ ($\theta$ = {theta_lbl})', fontsize=11)

    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(zlabel, fontsize=12)
    ax_main.set_title(
        'Trunk longitudinal profile colored by $k_{sn}$ with valley width', fontsize=13
    )
    ax_main.grid(True, alpha=0.3)

    # Twin axis: valley width (smoothed-on-s2 dots + MA curve + shading).
    ax_w = ax_main.twinx()
    valid_vw_dots = np.isfinite(vw_smooth_sorted)
    if valid_vw_dots.any():
        ax_w.scatter(
            xd_sorted_m[valid_vw_dots] * scale_x,
            vw_smooth_sorted[valid_vw_dots] * scale_w,
            c='steelblue',
            s=6,
            alpha=0.2,
            linewidths=0,
            edgecolors='none',
            zorder=2,
            label='Valley width (smoothed on s2)',
        )
    valid_vw_ma = np.isfinite(vw_ma_sorted)
    if valid_vw_ma.any():
        ax_w.fill_between(
            xd_sorted_m[valid_vw_ma] * scale_x,
            0.0,
            vw_ma_sorted[valid_vw_ma] * scale_w,
            color='steelblue',
            alpha=0.25,
            zorder=2,
        )
        ax_w.plot(
            xd_sorted_m[valid_vw_ma] * scale_x,
            vw_ma_sorted[valid_vw_ma] * scale_w,
            color='steelblue',
            linewidth=1.4,
            alpha=0.9,
            zorder=3,
            label=f'Valley width (MA, {ma_window_m:g} m)',
        )
    ax_w.set_ylabel(wlabel, fontsize=12, color='steelblue')
    ax_w.tick_params(axis='y', labelcolor='steelblue')
    ax_w.legend(loc='upper right', fontsize=10)
    if valleywidth_ylim is not None:
        ymin, ymax = valleywidth_ylim
        ax_w.set_ylim(ymin * scale_w, ymax * scale_w)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_chi_z_trunk_colored_gray_network(
    S_all,
    st_trunk,
    dem,
    acc,
    ksn_st_smoothed,
    vmin=0,
    vmax=None,
    theta=None,
    figsize=(10, 8),
    savefig=None,
):
    """All streams as light gray χ–Z points; trunk colored by smoothed k_sn.

    Match the Mataian left panel with ``st_trunk = S_all.trunk()``, fixed ``vmax`` (e.g. 600),
    ``vmin=0``, and ``theta`` equal to the value passed to ``calculate_ksn``.
    """
    s2_chi = S_all.chitransform(acc)
    s2_z = S_all.ezgetnal(dem)
    s2_z_chi = S_all.xy(data=(s2_z, s2_chi))

    all_z_gray = []
    all_chi_gray = []
    for z_chi in s2_z_chi:
        if not z_chi:
            continue
        z_vals, chi_vals = zip(*z_chi)
        all_z_gray.extend(z_vals)
        all_chi_gray.extend(chi_vals)
    all_z_gray = np.asarray(all_z_gray)
    all_chi_gray = np.asarray(all_chi_gray)

    st_chi = st_trunk.chitransform(acc)
    st_z = st_trunk.ezgetnal(dem)

    # Automatically set vmax if not provided
    ksn_vmax = vmax
    if vmax is None:
        ksn_vmax = np.nanmax(ksn_st_smoothed)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        all_chi_gray, all_z_gray, c='gray', s=1, alpha=0.3,
        linewidths=0, edgecolors='none', zorder=1,
    )
    sc = ax.scatter(
        st_chi, st_z, c=ksn_st_smoothed, cmap='viridis',
        s=8, vmin=vmin, vmax=ksn_vmax, alpha=0.7, linewidths=0, edgecolors='none', zorder=5,
    )
    if theta is not None:
        cbar_lbl = rf'$k_{{sn}}$ ($\theta$ = {abs(float(theta)):.2f})'
    else:
        cbar_lbl = r'$k_{sn}$'
    plt.colorbar(sc, ax=ax, pad=0.01, label=cbar_lbl)
    ax.set_xlabel(r'$\chi$ (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title(r'$\chi$–Z: trunk colored by $k_{sn}$', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def _trunk_flat_xy_chi_z_ksn(st, dem, acc, dem_imposed, theta, ksn_window):
    st_chi_arr = np.array(st.chitransform(acc))
    st_z_arr = np.array(st.ezgetnal(dem))
    st_xy_groups = st.xy()
    st_z_chi_groups = st.xy(data=(st_z_arr, st_chi_arr))
    st_ksn = calculate_ksn(st, dem_imposed, acc, theta)
    st_ksn_s = smooth_stream_values(st, st_ksn, window_size=ksn_window)

    trunk_x, trunk_y, trunk_chi, trunk_z, trunk_ksn = [], [], [], [], []
    idx = 0
    for coord_group, zc_group in zip(st_xy_groups, st_z_chi_groups):
        n = len(coord_group)
        for j, ((cx, cy), (zv, chiv)) in enumerate(zip(coord_group, zc_group)):
            trunk_x.append(cx)
            trunk_y.append(cy)
            trunk_chi.append(chiv)
            trunk_z.append(zv)
            trunk_ksn.append(st_ksn_s[idx + j])
        idx += n
    return (
        np.array(trunk_x), np.array(trunk_y), np.array(trunk_chi),
        np.array(trunk_z), np.array(trunk_ksn),
    )


def plot_chi_z_trunk_ksn_valley_width_twin(
    st,
    dem,
    acc,
    dem_imposed,
    allwidths,
    theta=-0.45,
    ksn_window=100,
    vw_window=100,
    max_match_dist=150.0,
    ksn_vmin=0,
    ksn_vmax=None,
    valleywidth_ylim=None,
    figsize=(16, 6),
    savefig=None,
):
    """χ–Z scatter colored by smoothed k_sn + twin axis stream-smoothed valley width vs χ."""
    trunk_x, trunk_y, trunk_chi, trunk_z, trunk_ksn = _trunk_flat_xy_chi_z_ksn(
        st, dem, acc, dem_imposed, theta, ksn_window
    )

    tree = cKDTree(np.column_stack([trunk_x, trunk_y]))
    vw_x = allwidths[:, 0]
    vw_y = allwidths[:, 1]
    vw_raw = allwidths[:, 2]
    dists, indices = tree.query(np.column_stack([vw_x, vw_y]))
    mask_close = dists < max_match_dist

    node_widths = defaultdict(list)
    for i in range(len(dists)):
        if dists[i] < max_match_dist:
            node_widths[indices[i]].append(vw_raw[i])

    vw_trunk_arr = np.full(len(trunk_x), np.nan)
    for flat_idx, widths in node_widths.items():
        vw_trunk_arr[flat_idx] = np.mean(widths)

    vw_filled = fill_nan_stream_segments(st, vw_trunk_arr)
    vw_trunk_smooth = smooth_stream_values(st, vw_filled, window_size=vw_window)

    sort_idx = np.argsort(trunk_chi)
    vw_chi_sorted = trunk_chi[sort_idx]
    vw_width_chi_smooth = vw_trunk_smooth[sort_idx]

    print(f'Valley width points matched to trunk: {mask_close.sum()} / {len(dists)}')

    # Automatically set ksn_vmax if not provided
    _ksn_vmax = ksn_vmax
    if ksn_vmax is None:
        _ksn_vmax = np.nanmax(trunk_ksn)

    fig, ax_chi = plt.subplots(figsize=figsize)
    sc1 = ax_chi.scatter(
        trunk_chi, trunk_z, c=trunk_ksn, cmap='viridis',
        s=8, vmin=ksn_vmin, vmax=_ksn_vmax, linewidths=0, edgecolors='none',
        zorder=3, alpha=0.9,
    )
    cbar1 = plt.colorbar(sc1, ax=ax_chi, pad=0.08)
    cbar1.set_label(r'$k_{sn}$ ($\theta$ = 0.45)', fontsize=11)
    ax_chi.set_xlabel(r'$\chi$ (m)', fontsize=12)
    ax_chi.set_ylabel('Elevation (m)', fontsize=12)
    ax_chi.set_title(r'$\chi$–Z colored by $k_{sn}$ with stream-smoothed valley width', fontsize=13)
    ax_chi.grid(True, alpha=0.3)

    ax2 = ax_chi.twinx()
    ax2.fill_between(vw_chi_sorted, 0, vw_width_chi_smooth, color='steelblue', alpha=0.25, zorder=1)
    ax2.plot(
        vw_chi_sorted, vw_width_chi_smooth, color='steelblue', linewidth=1.2, alpha=0.8, zorder=2,
        label='Stream-smoothed valley width',
    )
    ax2.set_ylabel('Valley Width (m)', fontsize=12, color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.legend(loc='upper right', fontsize=10)
    if valleywidth_ylim is not None:
        ax2.set_ylim(valleywidth_ylim)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_threshold_and_excess_topography(
    dem,
    slope_degrees=35.0,
    figsize=(15, 12),
    savefig=None,
    basin_mask=None,
):
    """Threshold (``fsm2d``) topography and DEM minus threshold surface for one slope angle.

    Parameters
    ----------
    dem : topotoolbox.GridObject
    slope_degrees : float
        Critical hillslope angle in degrees (converted to tangent threshold).
    basin_mask : ndarray of bool, optional
        If given, excess panel masks *outside* the basin (``~basin_mask`` as NaN), matching
        the Mataian notebook style. If None, the full grid is shown.
    """
    threshold_S = float(np.tan(np.radians(slope_degrees)))
    excess = dem.excesstopography(method='fsm2d', threshold=threshold_S)

    bounds = dem.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    excess_arr = np.asarray(excess)
    vmin = np.nanmin(excess_arr)
    vmax = np.nanmax(excess_arr)
    excess.plot_hs(ax=ax1, cmap='gist_earth', vmin=vmin, vmax=vmax)
    if basin_mask is not None:
        gray_overlay = excess_arr.copy()
        gray_overlay[basin_mask] = np.nan
        ax1.imshow(
            gray_overlay, cmap='gray', alpha=0.7, extent=extent,
            vmin=vmin, vmax=vmax, origin='upper',
        )
    ax1.set_title(f'Threshold topography (fsm2d, {slope_degrees:.0f}°)')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.images[0]
    plt.colorbar(im1, cax=cax1, label='Elevation (m)')

    diff = dem - excess
    diff_arr = np.array(diff)
    diff_plot = diff_arr.copy()
    if basin_mask is not None:
        diff_plot[~basin_mask] = np.nan
    p_lo, p_hi = np.nanpercentile(diff_arr, [0, 99])
    dem.plot_hs(ax=ax2, cmap='gray')
    im2 = ax2.imshow(
        diff_plot, cmap='viridis', alpha=0.75, extent=extent,
        norm=mcolors.Normalize(vmin=p_lo, vmax=p_hi), origin='upper',
    )
    ax2.set_title(f'Excess topography (DEM − threshold surface, {slope_degrees:.0f}°)')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Excess elevation (m)')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()
