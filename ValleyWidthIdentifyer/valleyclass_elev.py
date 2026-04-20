"""Valley classification based on elevation threshold above nearest stream.

Translates the MATLAB function ``valleyclass_elev.m`` from the
ElevationThresholdValleyWidth toolbox.
"""

import numpy as np
import copy


def valleyclass_elev(dem, stream, flow, elevthreshold, plot=False,
                     nodata_mask=None):
    """Classify each DEM pixel as valley (1), stream (2), or hillslope (0).

    Uses height-above-nearest-drainage (HAND) computed via
    ``FlowObject.vertdistance2stream`` and a vertical elevation threshold.

    Parameters
    ----------
    dem : topotoolbox.GridObject
        Digital elevation model.
    stream : topotoolbox.StreamObject
        Stream network derived from the DEM.
    flow : topotoolbox.FlowObject
        Flow direction object derived from the DEM.
    elevthreshold : float
        Maximum vertical distance (m) above the nearest stream for a
        pixel to be classified as valley.
    plot : bool, optional
        If True, display three separate diagnostic figures (DEM + streams with
        elevation colorbar, classification, HAND). Default is False.
    nodata_mask : np.ndarray of bool, optional
        Boolean array with the same shape as ``dem`` marking NoData/empty
        pixels. When provided, these pixels are forced to hillslope (0) in
        the output classification and their HAND is set to NaN in diagnostic
        plots. ``None`` (default) preserves the original behavior.

    Returns
    -------
    topotoolbox.GridObject
        Valley classification grid with the same dimensions as *dem*.
        Pixel values: 0 = hillslope, 1 = valley, 2 = stream channel.
    """
    print("Setting up the DEM")

    # Height above nearest drainage
    print("Finding DZ (vertical distance to stream)")
    DZ = flow.vertdistance2stream(stream, dem)
    dz_arr = np.asarray(DZ.z, dtype=np.float64)

    # Build classification array
    dv_arr = np.zeros(dem.shape, dtype=np.float32)

    # Mark stream pixels
    rows, cols = stream.node_indices
    dv_arr[rows, cols] = 2.0

    # Mark valley pixels where HAND < threshold (and not already stream)
    valley_mask = (dz_arr < elevthreshold) & (dv_arr < 2.0)
    dv_arr[valley_mask] = 1.0

    # Force NoData pixels to hillslope so they don't register as valley
    if nodata_mask is not None:
        dv_arr[nodata_mask] = 0.0
        dz_arr[nodata_mask] = np.nan
        DZ.z = dz_arr

    # Create output GridObject (copy of dem with replaced z)
    DV = copy.deepcopy(dem)
    DV.z = dv_arr

    if plot:
        _plot_valley(dem, DV, DZ, stream)

    print("Valley classification complete")
    return DV


def _plot_valley(dem, DV, DZ, stream):
    """Diagnostic plots for valley classification (one figure per panel)."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    bounds = dem.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    dv_arr = np.asarray(DV.z)
    dz_arr = np.asarray(DZ.z)
    dem_array = np.asarray(dem, dtype=float)
    dem_norm = colors.Normalize(
        vmin=np.nanmin(dem_array), vmax=np.nanmax(dem_array)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    dem.plot_hs(ax=ax, cmap='gist_earth', norm=dem_norm)
    stream.plot(ax=ax, color='w', linewidth=0.7, alpha=0.95)
    stream.plot(ax=ax, color='b', linewidth=1.0)
    ax.set_title("DEM with streams")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.05)
    elev_sm = ScalarMappable(norm=dem_norm, cmap='gist_earth')
    elev_sm.set_array(dem_array)
    fig.colorbar(elev_sm, cax=cax, label='Elevation (m)')

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('viridis', 3)
    im = ax.imshow(
        dv_arr, extent=extent, origin='upper',
        cmap=cmap, vmin=-0.5, vmax=2.5,
    )
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2],
                 label='0=hillslope, 1=valley, 2=stream')
    ax.set_title("Valley classification")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        dz_arr, extent=extent, origin='upper',
        cmap='RdYlBu_r', vmin=0, vmax=np.nanpercentile(dz_arr, 95),
    )
    plt.colorbar(im, ax=ax, label='Height above stream (m)')
    stream.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("HAND")
    fig.tight_layout()
    plt.show()
