"""End-to-end pipeline: DEM to valley widths.

Translates the MATLAB function ``DEM2widths.m`` from the
ElevationThresholdValleyWidth toolbox.
"""

import numpy as np
from scipy.spatial import cKDTree

from .valleyclass_elev import valleyclass_elev
from .stream_utils import removeshortstreams, stream2swath
from .swath_width import swath_width


def dem2widths(dem, streamarea, elevthreshold, swath_dx, minradius,
              swath_width_param, units='pixels', plot=False):
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
    swath_width_param : float
        Total width (m) of cross-stream profiles.
    units : str, optional
        Units for *streamarea*: ``'pixels'`` (default), ``'m2'``, etc.
    plot : bool, optional
        If True, show diagnostic plots during valley classification.

    Returns
    -------
    allwidths : np.ndarray
        (N x 6) array. Columns: x, y, raw_width, min_width,
        drainage_area (m^2), stream_gradient (degrees).
    DV : topotoolbox.GridObject
        Valley classification grid (0 = hillslope, 1 = valley, 2 = stream).
    """
    import topotoolbox as topo

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

    # --- Valley classification ---
    print("Classifying valley")
    DV = valleyclass_elev(dem, S, FD, elevthreshold, plot=plot)

    # --- Swath profiles ---
    print("Swath width extraction")
    swath_profiles = stream2swath(S, DV, swath_dx, swath_width_param)
    allwidths = swath_width(swath_profiles, minradius)

    if allwidths.shape[0] == 0:
        print("Warning: no width measurements produced.")
        return allwidths, DV

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
    _, idx = tree.query(allwidths[:, :2])
    allwidth_DAs = S_A_m[idx]
    allwidths = np.column_stack([allwidths, allwidth_DAs])

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
    allwidths = np.column_stack([allwidths, allwidth_grads])

    print(f"Done. {allwidths.shape[0]} width measurements extracted.")
    return allwidths, DV
