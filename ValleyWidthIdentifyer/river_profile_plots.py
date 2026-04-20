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

    Uses the same ``streamquad_trapz_f32`` integration as ``plotdz``/Mataian long-profile
    code, but returns one point per node so branches plot as a point cloud instead of
    separate polylines.
    """
    from topotoolbox import _stream

    s_z = S_all.ezgetnal(dem)
    dist_m = np.zeros_like(s_z, dtype=np.float32)
    one = np.ones_like(s_z, dtype=np.float32)
    _stream.streamquad_trapz_f32(
        dist_m, one, S_all.source, S_all.target, S_all.distance()
    )
    x_km = np.asarray(dist_m / 1000.0, dtype=np.float64)
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

    from topotoolbox import _stream

    # Compute all stream nodes' distances (network, not polylines)
    s2_z = S_all.ezgetnal(dem)
    dist_s2_m = np.zeros_like(s2_z, dtype=np.float32)
    one = np.ones_like(s2_z, dtype=np.float32)
    _stream.streamquad_trapz_f32(
        dist_s2_m, one, S_all.source, S_all.target, S_all.distance()
    )
    all_dist_km = np.asarray(dist_s2_m / 1000.0, dtype=np.float64)
    all_z = np.asarray(s2_z, dtype=np.float64)

    # Branches: panel-style as in plot_chi_z_trunk_colored_gray_network
    # Plot all branches as faint gray points (excluding trunk)
    st_trunk_nodes = set(st_trunk.nodes) if hasattr(st_trunk, 'nodes') else set()
    if not st_trunk_nodes:
        # If not present, try to recover from S_all.trunk()
        st_trunk_nodes = set(getattr(S_all.trunk(), 'nodes', []))
    # Compose per-branch node indices via xy
    s2_z = S_all.ezgetnal(dem)
    s2_dist = np.asarray(dist_s2_m / 1000.0, dtype=np.float64)
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

    # Trunk
    st_z = st_trunk.ezgetnal(dem)
    dist_trunk = np.zeros_like(st_z, dtype=np.float32)
    a = np.ones_like(st_z, dtype=np.float32)
    _stream.streamquad_trapz_f32(
        dist_trunk, a, st_trunk.source, st_trunk.target, st_trunk.distance()
    )
    trunk_dist_km = dist_trunk / 1000.0

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
    vw_window=100,
    max_match_dist=150.0,
    vw_outline_sigma_pts=4.0,
    ksn_vmin=0,
    ksn_vmax=None,
    ksn_clim=None,
    valleywidth_ylim=None,
    figsize=(16, 6),
    savefig=None,
):
    """Long profile (distance–Z): gray scatter of all *s2* nodes, trunk colored by k_sn, twin = width.

    Gray background uses the same outlet-distance vs elevation point cloud as the Mataian
    all-stream long profile (not ``plotdz`` polylines). Valley width uses the **same**
    smoothed-on-*s2* field as the map pipeline, sampled onto trunk NAL nodes via
    ``trunk_valley_width_from_s2_field``, then drawn in ``st.xy`` segment order with a light
    Gaussian on the outline.
    """
    from topotoolbox import _stream

    st_z = np.asarray(st.ezgetnal(dem), dtype=float)
    dist_trunk = np.zeros_like(st_z, dtype=np.float32)
    a_ones = np.ones_like(st_z, dtype=np.float32)
    _stream.streamquad_trapz_f32(dist_trunk, a_ones, st.source, st.target, st.distance())
    trunk_dist_km = dist_trunk / 1000.0

    ksn_st = calculate_ksn(st, dem_imposed, acc, theta)
    ksn_st_smoothed = smooth_stream_values(st, ksn_st, window_size=ksn_window)

    s2_vw_bundle = valley_width_smoothed_on_stream(
        allwidths, s2, max_match_dist=max_match_dist, window_size=vw_window
    )
    s2_coords_flat, s2_vw_smooth_flat, dists_aw, _ = s2_vw_bundle
    vw_trunk_smooth_nal = trunk_valley_width_from_s2_field(
        allwidths,
        s2,
        st,
        max_match_dist=max_match_dist,
        window_size=vw_window,
        s2_field=(s2_coords_flat, s2_vw_smooth_flat),
    )

    min_len = min(
        len(trunk_dist_km), len(st_z), len(ksn_st_smoothed), len(vw_trunk_smooth_nal)
    )
    trunk_dist_km = np.asarray(trunk_dist_km[:min_len], dtype=np.float64)
    st_z = st_z[:min_len]
    ksn_st_smoothed = ksn_st_smoothed[:min_len]
    vw_trunk_smooth_nal = np.asarray(vw_trunk_smooth_nal[:min_len], dtype=np.float64)

    # ksn_clim (vmin, vmax) overrides ksn_vmin/ksn_vmax when provided.
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
    print(f'Valley width points matched to s2 network: {mask_close.sum()} / {len(dists_aw)}')
    print('Trunk overlay: s2 field; width line uses st.xy segment order (no global distance sort)')

    seg_d_w = st.xy(
        data=(
            np.asarray(trunk_dist_km, dtype=np.float64),
            np.asarray(vw_trunk_smooth_nal, dtype=np.float64),
        )
    )

    theta_lbl = f'{abs(theta):.2f}'

    fig, ax_main = plt.subplots(figsize=figsize)

    # --- REWRITE of gray branches plotting to follow the style of plot_longitudinal_trunk_colored_ksn and plot_chi_z_trunk_colored_gray_network ---

    # Compute all stream nodes' distances (network, not polylines)
    from topotoolbox import _stream

    s2_z = s2.ezgetnal(dem)
    dist_s2_m = np.zeros_like(s2_z, dtype=np.float32)
    one = np.ones_like(s2_z, dtype=np.float32)
    _stream.streamquad_trapz_f32(
        dist_s2_m, one, s2.source, s2.target, s2.distance()
    )
    all_dist_km = np.asarray(dist_s2_m / 1000.0, dtype=np.float64)
    all_z = np.asarray(s2_z, dtype=np.float64)

    # Plot all branches as faint gray points (excluding trunk)
    st_trunk_nodes = set(st.nodes) if hasattr(st, 'nodes') else set()
    if not st_trunk_nodes:
        st_trunk_nodes = set(getattr(s2.trunk(), 'nodes', []))
    s2_z = s2.ezgetnal(dem)
    s2_dist = np.asarray(dist_s2_m / 1000.0, dtype=np.float64)
    groups = s2.xy(data=(s2_dist, s2_z))
    background_dist = []
    background_z = []
    for seg in groups:
        if not seg:
            continue
        is_trunk_seg = False
        if hasattr(s2, "nodes") and hasattr(st, "nodes"):
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
        background_dist,
        background_z,
        c='gray',
        s=1,
        alpha=0.3,
        linewidths=0,
        edgecolors='none',
        zorder=1,
    )

    # --- END REWRITE gray branch plotting ---

    sc1 = ax_main.scatter(
        trunk_dist_km,
        st_z,
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

    ax_main.set_xlabel('Distance (km)', fontsize=12)
    ax_main.set_ylabel('Elevation (m)', fontsize=12)
    ax_main.set_title(
        'Trunk longitudinal profile colored by $k_{sn}$ with valley width', fontsize=13
    )
    ax_main.grid(True, alpha=0.3)

    ax_w = ax_main.twinx()
    first_leg = True
    for seg in seg_d_w:
        if len(seg) < 1:
            continue
        arr = np.asarray(seg, dtype=np.float64)
        o = np.argsort(arr[:, 0])
        xd = arr[o, 0]
        yw = arr[o, 1]
        if yw.size >= 5:
            yw = gaussian_filter1d(yw, sigma=vw_outline_sigma_pts, mode='nearest')
        elif yw.size >= 3:
            yw = gaussian_filter1d(
                yw, sigma=min(vw_outline_sigma_pts, yw.size / 3.0), mode='nearest'
            )
        pkw = dict(color='steelblue', linewidth=1.2, alpha=0.8, zorder=3)
        if first_leg:
            pkw['label'] = 'Valley width (s2 field, trunk fill)'
            first_leg = False
        ax_w.fill_between(xd, 0, yw, color='steelblue', alpha=0.25, zorder=2)
        ax_w.plot(xd, yw, **pkw)
    ax_w.set_ylabel('Valley Width (m)', fontsize=12, color='steelblue')
    ax_w.tick_params(axis='y', labelcolor='steelblue')
    ax_w.legend(loc='upper right', fontsize=10)
    if valleywidth_ylim is not None:
        ax_w.set_ylim(valleywidth_ylim)

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
