"""Valley classification based on elevation threshold above nearest stream.

Translates the MATLAB function ``valleyclass_elev.m`` from the
ElevationThresholdValleyWidth toolbox.
"""

import numpy as np
import copy


def valleyclass_elev(dem, stream, flow, elevthreshold, plot=False):
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
        If True, display diagnostic plots. Default is False.

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

    # Create output GridObject (copy of dem with replaced z)
    DV = copy.deepcopy(dem)
    DV.z = dv_arr

    if plot:
        _plot_valley(dem, DV, DZ, stream)

    print("Valley classification complete")
    return DV


def _plot_valley(dem, DV, DZ, stream):
    """Diagnostic plots for valley classification."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # DEM hillshade
    ax = axes[0]
    dem.plot_hs(ax=ax, cmap='gist_earth',
                norm=colors.Normalize(vmin=0, vmax=np.nanmax(np.asarray(dem))))
    stream.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("DEM with streams")

    # Valley classification
    ax = axes[1]
    dv_arr = np.asarray(DV.z)
    bounds = dem.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    cmap = plt.cm.get_cmap('viridis', 3)
    im = ax.imshow(dv_arr, extent=extent, origin='upper',
                   cmap=cmap, vmin=-0.5, vmax=2.5)
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2],
                 label='0=hillslope, 1=valley, 2=stream')
    ax.set_title("Valley classification")

    # Vertical distance to stream
    ax = axes[2]
    dz_arr = np.asarray(DZ.z)
    im = ax.imshow(dz_arr, extent=extent, origin='upper',
                   cmap='RdYlBu_r', vmin=0, vmax=np.nanpercentile(dz_arr, 95))
    plt.colorbar(im, ax=ax, label='Height above stream (m)')
    stream.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("HAND")

    plt.tight_layout()
    plt.show()
