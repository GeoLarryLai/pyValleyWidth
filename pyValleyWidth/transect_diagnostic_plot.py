"""Diagnostic plot for the per-transect spillover-rim detection.

Provides one public function, :func:`plot_transect_diagnostic`, that
generates a clean three-panel figure for a single cross-section:

* a map (left, full-height) showing the DEM, stream, and the transect
  line at the chosen profile location;
* the HAND profile of that cross-section (right, top) with the picked
  rim threshold and the measured valley filled in;
* the width-sensitivity sweep (right, bottom) showing Width(T) and
  dW/dT with the picked peak marked.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec

from .stream_utils import stream2swath
from .peak_valley_width import _width_sensitivity_threshold


def _auto_swath_width(dem, stream):
    """Distance-to-divide based swath width (q0.99 of distance from stream)."""
    from scipy.ndimage import distance_transform_edt

    rows, cols = stream.node_indices
    smask = np.zeros(dem.shape, dtype=bool)
    smask[rows, cols] = True
    dist_m = distance_transform_edt(~smask) * dem.cellsize
    valid = dist_m[dist_m > 0]
    if valid.size == 0:
        return 200.0 * dem.cellsize
    cs2 = 2.0 * dem.cellsize
    q = float(np.quantile(valid, 0.99))
    return float(np.ceil(2.0 * q / cs2) * cs2)


def _build_hand(dem, fd, stream):
    """Compute HAND for *stream* and zero NoData pixels to NaN."""
    DZ = fd.vertdistance2stream(stream, dem)
    da = np.asarray(dem.z)
    nm = ~np.isfinite(da)
    for attr in ('nodata', '_nodata', 'nodataval'):
        v = getattr(dem, attr, None)
        if v is not None and np.isfinite(v):
            nm |= (da == float(v))
            break
    if nm.any():
        z = np.asarray(DZ.z, dtype=np.float64)
        z[nm] = np.nan
        DZ.z = z
    return DZ


def _flatten_profiles(swaths, cellsize):
    """Flatten a list of SwathProfile into per-profile dicts."""
    profs = []
    for sw in swaths:
        if sw.disty.size > 1:
            ds = abs(float(np.median(np.diff(sw.disty))))
        else:
            ds = float(cellsize)
        for pp in range(sw.xy.shape[0]):
            profs.append(dict(
                hand=sw.Z[:, pp],
                disty=sw.disty,
                xy=sw.xy[pp],
                ds=ds,
            ))
    return profs


def _transect_normal(profs, idx):
    """Unit normal (perpendicular to stream) at profile *idx*."""
    i0 = max(idx - 1, 0)
    i1 = min(idx + 1, len(profs) - 1)
    tang = profs[i1]['xy'] - profs[i0]['xy']
    n = np.array([-tang[1], tang[0]], dtype=np.float64)
    L = np.linalg.norm(n)
    return n / L if L > 0 else np.array([0.0, 1.0])


def plot_transect_diagnostic(
    dem,
    fd,
    stream,
    xy,
    *,
    min_dwdt=0.0,
    max_dwdt=None,
    clip_to_divides=True,
    swath_width='auto',
    swath_dx=None,
    hand=None,
    title=None,
    color='#377eb8',
    figsize=(16, 6),
    save_fig=False,
    filename=None,
    dpi=150,
):
    """Three-panel diagnostic figure for one cross-section.

    The function snaps the user-supplied map point ``xy`` to the
    nearest profile centre on ``stream``, runs the spillover-rim
    detection (``min_dwdt`` / ``max_dwdt``) on that profile, and
    renders a clean three-panel figure.

    Layout (GridSpec 2x2, width_ratios=[1.3, 1])::

        +----------------+--------------------+
        |                |  HAND profile      |
        |  Map (full     +--------------------+
        |  height)       |  Width(T) + dW/dT  |
        +----------------+--------------------+

    Parameters
    ----------
    dem : topotoolbox.GridObject
        DEM grid (used for hillshade and HAND computation).
    fd : topotoolbox.FlowObject
        Flow object (used to compute HAND when ``hand`` is None).
    stream : topotoolbox.StreamObject
        Stream network to sample (e.g. ``S.trunk()``).
    xy : tuple of float
        ``(x, y)`` map coordinates of the desired cross-section.  The
        function snaps to the nearest profile centre.
    min_dwdt : float, optional
        Absolute floor (m/m) on dW/dT for a peak to qualify.
        Default 0.0.
    max_dwdt : float or None, optional
        Absolute ceiling (m/m) on dW/dT.  Peaks above this mark a
        divide-scale breakout.  Default None.
    clip_to_divides : bool, optional
        If True (default), the cross-section is clipped to its local
        drainage divides (max-HAND positions on each side of the
        channel) before the width sweep, plot, and width measurement.
    swath_width : float or 'auto', optional
        Total cross-stream transect length (m).  ``'auto'`` (default)
        uses 2 x q0.99 of the distance-to-stream raster.
    swath_dx : float, optional
        Along-stream profile spacing (m).  Defaults to ``dem.cellsize``.
    hand : topotoolbox.GridObject, optional
        Precomputed HAND grid for ``stream``.  When None, computed
        internally via ``fd.vertdistance2stream``.
    title : str, optional
        Title rendered above the map panel.
    color : str, optional
        Color used for the transect line and the profile dot on the
        map.  Default ``'#377eb8'``.
    figsize : tuple of float, optional
        Figure size (inches).  Default ``(16, 6)``.
    save_fig : bool, optional
        If True, save the figure via ``fig.savefig(filename, dpi=dpi)``.
    filename : str, optional
        Output path including extension (.png, .svg, .jpg, .pdf).
        Required when ``save_fig=True``.
    dpi : int, optional
        DPI for saved raster output.  Default 150.

    Returns
    -------
    fig : matplotlib.figure.Figure
    info : dict
        Dict with the picked profile index, snapped xy, and the
        ``_width_sensitivity_threshold`` result (or None).
    """
    if hand is None:
        hand_grid = _build_hand(dem, fd, stream)
    else:
        hand_grid = hand

    sw_w = _auto_swath_width(dem, stream) if swath_width == 'auto' else float(swath_width)
    s_dx = float(dem.cellsize) if swath_dx is None else float(swath_dx)

    swaths = stream2swath(stream, hand_grid, s_dx, sw_w)
    profs = _flatten_profiles(swaths, dem.cellsize)
    if not profs:
        raise RuntimeError('stream2swath produced no profiles for the given stream.')

    xy_arr = np.array([p['xy'] for p in profs], dtype=np.float64)
    target = np.asarray(xy, dtype=np.float64)
    si = int(np.argmin(np.sum((xy_arr - target) ** 2, axis=1)))

    p = profs[si]
    dy, hand_profile, ds = p['disty'], p['hand'], p['ds']

    ws = _width_sensitivity_threshold(
        hand_profile, dy, ds,
        min_dwdt=min_dwdt,
        max_dwdt=max_dwdt,
        clip_to_divides=clip_to_divides,
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1.0],
                  height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    dem_norm = mcolors.Normalize(vmin=np.nanmin(np.asarray(dem)),
                                 vmax=np.nanmax(np.asarray(dem)))
    dem.plot_hs(ax=ax_map, cmap='gist_earth', norm=dem_norm)
    stream.plot(ax=ax_map, color='cyan', linewidth=0.8)

    n_dir = _transect_normal(profs, si)
    end1 = p['xy'] + (sw_w / 2.0) * n_dir
    end2 = p['xy'] - (sw_w / 2.0) * n_dir
    ax_map.plot([end1[0], end2[0]], [end1[1], end2[1]], '-',
                color=color, lw=1.6, alpha=0.9)
    ax_map.plot(p['xy'][0], p['xy'][1], 'o',
                color=color, ms=11, mec='k', mew=1.2, zorder=5)
    if title:
        ax_map.set_title(title, fontsize=12, color=color, fontweight='bold')
    ax_map.set_xlabel('Easting (m)', fontsize=10)
    ax_map.set_ylabel('Northing (m)', fontsize=10)

    if ws is not None and clip_to_divides:
        divide_l, divide_r = ws['divide_l'], ws['divide_r']
        in_drainage = (dy >= divide_l) & (dy <= divide_r)
        hand_display = np.where(in_drainage, hand_profile, np.nan)
    else:
        divide_l = float(dy[0])
        divide_r = float(dy[-1])
        hand_display = hand_profile

    ax_top.plot(dy, hand_display, 'k-', lw=1.0, alpha=0.65, label='HAND')
    ax_top.axvline(0, color='cyan', lw=1.5, ls='--', alpha=0.7)

    if ws is not None:
        thr = ws['thr']
        edge_l, edge_r = ws['edge_l'], ws['edge_r']
        vw = edge_r - edge_l

        ax_top.axhline(thr, color='green', lw=2.0, alpha=0.85,
                       label=f'Rim T = {thr:.0f} m')

        ok = np.isfinite(hand_display)
        in_valley = (dy >= edge_l) & (dy <= edge_r) & ok & (hand_display < thr)
        ax_top.fill_between(dy, 0, np.where(ok, hand_display, 0),
                            where=in_valley, alpha=0.2, color='green',
                            label=f'Valley width ≈ {vw:.0f} m')

        ax_top.axvline(edge_l, color='green', lw=1, ls=':', alpha=0.7)
        ax_top.axvline(edge_r, color='green', lw=1, ls=':', alpha=0.7)

        ax_top.axvline(divide_l, color='gray', lw=1, ls='--', alpha=0.7,
                       label='Divide')
        ax_top.axvline(divide_r, color='gray', lw=1, ls='--', alpha=0.7)

        margin = max(0.05 * (divide_r - divide_l), 50.0)
        xlim = (divide_l - margin, divide_r + margin)
    else:
        xlim = (-500.0, 500.0)

    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(bottom=0)
    ax_top.set_xlabel('Cross-stream distance (m)', fontsize=10)
    ax_top.set_ylabel('HAND (m)', fontsize=10)
    ax_top.legend(fontsize=9, loc='upper right')
    ax_top.grid(True, alpha=0.3)

    if ws is not None:
        ax_w = ax_bot
        ax_dw = ax_bot.twinx()

        ax_w.plot(ws['thresholds'], ws['widths_s'], 'b-', lw=1.2,
                  label='Width(T)')
        ax_w.set_xlabel('HAND threshold (m)', fontsize=10)
        ax_w.set_ylabel('Width (m)', fontsize=10, color='b')
        ax_w.tick_params(axis='y', labelcolor='b')

        ax_dw.plot(ws['thresholds'], ws['dw'], 'r-', lw=1.0, alpha=0.75,
                   label='dW/dT')
        ax_dw.set_ylabel('dW/dT (m/m)', fontsize=10, color='r')
        ax_dw.tick_params(axis='y', labelcolor='r')

        if max_dwdt is not None:
            ax_dw.axhline(max_dwdt, color='r', lw=0.8, ls=':', alpha=0.5)
        if min_dwdt > 0:
            ax_dw.axhline(min_dwdt, color='r', lw=0.8, ls=':', alpha=0.5)

        ax_w.axvline(ws['thr'], color='green', lw=2.0, alpha=0.85)
        ax_dw.plot(ws['thr'], ws['dw'][ws['rim_idx']], '*',
                   color='green', ms=13, mec='k', mew=0.8, zorder=6)

        h_w, l_w = ax_w.get_legend_handles_labels()
        h_d, l_d = ax_dw.get_legend_handles_labels()
        ax_w.legend(h_w + h_d, l_w + l_d, fontsize=9, loc='upper left')
        ax_w.grid(True, alpha=0.3)
    else:
        ax_bot.text(0.5, 0.5, 'Rim detection failed',
                    ha='center', va='center', transform=ax_bot.transAxes,
                    fontsize=11, color='gray')
        ax_bot.set_axis_off()

    if save_fig:
        if not filename:
            raise ValueError('save_fig=True requires a filename.')
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        if ext not in {'png', 'svg', 'jpg', 'jpeg', 'pdf'}:
            raise ValueError(
                f"Unsupported figure format '{ext}'. "
                "Use one of: png, svg, jpg, pdf.")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    return fig, dict(profile_idx=si, snapped_xy=p['xy'].copy(), ws=ws)
