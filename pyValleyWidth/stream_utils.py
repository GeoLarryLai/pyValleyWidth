"""Bridge functions for MATLAB TopoToolbox 2 features missing from Python topotoolbox.

Provides:
    removeshortstreams - Remove stream segments shorter than a threshold length.
    stream2swath       - Generate cross-stream swath profiles (replaces STREAMobj2SWATHobj).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import List


@dataclass
class SwathProfile:
    """Container for swath profile data along a single stream segment.

    Attributes
    ----------
    Z : np.ndarray
        2D array (n_perp_samples x n_profiles) of sampled grid values.
    disty : np.ndarray
        1D array of perpendicular distances from the stream center.
    xy : np.ndarray
        (n_profiles x 2) array of profile center (x, y) coordinates.
    width : float
        Total cross-profile width in map units.
    """
    Z: np.ndarray
    disty: np.ndarray
    xy: np.ndarray
    width: float


def removeshortstreams(stream, minlength):
    """Remove stream segments shorter than *minlength* (map units).

    Parameters
    ----------
    stream : topotoolbox.StreamObject
        Stream network.
    minlength : float
        Minimum stream length in map units (e.g. metres).

    Returns
    -------
    topotoolbox.StreamObject
        Pruned stream network with short segments removed.
    """
    n_nodes = stream.stream.size

    # Build adjacency from edges (undirected for connected components)
    adj = [[] for _ in range(n_nodes)]
    for s, t in zip(stream.source, stream.target):
        adj[s].append(t)
        adj[t].append(s)

    # Get geographic coordinates of every node
    rows, cols = stream.node_indices
    xs = stream.transform[0] * cols + stream.transform[2]
    ys = stream.transform[4] * rows + stream.transform[5]

    # Find connected components via BFS and compute total length
    visited = np.zeros(n_nodes, dtype=bool)
    keep = np.zeros(n_nodes, dtype=bool)

    for start in range(n_nodes):
        if visited[start]:
            continue
        # BFS to collect component
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        # Compute total edge length within this component
        comp_set = set(component)
        total_length = 0.0
        for s, t in zip(stream.source, stream.target):
            if s in comp_set and t in comp_set:
                dx = xs[s] - xs[t]
                dy = ys[s] - ys[t]
                total_length += np.sqrt(dx * dx + dy * dy)

        if total_length >= minlength:
            keep[list(comp_set)] = True

    if not np.any(keep):
        import warnings
        warnings.warn(
            f"All streams are shorter than {minlength}. "
            "Returning original stream unchanged.")
        return stream

    return stream.subgraph(keep)


def _resample_polyline(coords, dx):
    """Resample a polyline at equal spacing *dx* along its length.

    Returns resampled (x, y) arrays and tangent unit vectors at each point.
    """
    coords = np.asarray(coords, dtype=np.float64)
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_dist[-1]

    if total_length < dx:
        mid = coords[len(coords) // 2]
        tang = diffs[0] if len(diffs) > 0 else np.array([1.0, 0.0])
        norm = np.sqrt(tang[0] ** 2 + tang[1] ** 2)
        tang = tang / max(norm, 1e-12)
        return mid.reshape(1, 2), tang.reshape(1, 2)

    sample_dists = np.arange(0, total_length, dx)
    if len(sample_dists) == 0:
        sample_dists = np.array([0.0])

    sample_x = np.interp(sample_dists, cum_dist, coords[:, 0])
    sample_y = np.interp(sample_dists, cum_dist, coords[:, 1])

    # Tangent vectors via finite differences on the interpolated points
    tangents = np.zeros((len(sample_dists), 2))
    if len(sample_dists) > 1:
        tangents[0] = [sample_x[1] - sample_x[0], sample_y[1] - sample_y[0]]
        tangents[-1] = [sample_x[-1] - sample_x[-2], sample_y[-1] - sample_y[-2]]
        for i in range(1, len(sample_dists) - 1):
            tangents[i] = [sample_x[i + 1] - sample_x[i - 1],
                           sample_y[i + 1] - sample_y[i - 1]]
    else:
        tangents[0] = diffs[0] if len(diffs) > 0 else [1.0, 0.0]

    norms = np.sqrt(tangents[:, 0] ** 2 + tangents[:, 1] ** 2)
    norms[norms < 1e-12] = 1e-12
    tangents /= norms[:, None]

    pts = np.column_stack([sample_x, sample_y])
    return pts, tangents


def stream2swath(stream, grid, dx, width):
    """Generate cross-stream swath profiles for a grid along a stream network.

    Replaces MATLAB ``STREAMobj2SWATHobj(S, DV, 'dx', dx, 'width', width)``.

    Parameters
    ----------
    stream : topotoolbox.StreamObject
        Stream network.
    grid : topotoolbox.GridObject
        Grid to sample (e.g. valley classification).
    dx : float
        Spacing between cross-profiles along the stream (map units).
    width : float
        Total cross-profile width (map units). Profiles extend width/2
        on each side of the stream.

    Returns
    -------
    list of SwathProfile
        One SwathProfile per stream segment.
    """
    half_width = width / 2.0
    cellsize = grid.cellsize

    # Number of perpendicular sample points on each side
    n_perp_half = int(np.ceil(half_width / cellsize))
    perp_dists = np.linspace(-half_width, half_width,
                             2 * n_perp_half + 1)

    # Build interpolator for the grid
    grid_z = np.asarray(grid.z, dtype=np.float64)
    nrows, ncols = grid_z.shape
    # Row/col centres in geographic coordinates
    t = grid.transform
    col_coords = t[0] * np.arange(ncols) + t[2]
    row_coords = t[4] * np.arange(nrows) + t[5]

    # RegularGridInterpolator expects ascending axes
    row_ascending = row_coords[0] < row_coords[-1]
    if not row_ascending:
        row_coords_interp = row_coords[::-1]
        grid_z_interp = grid_z[::-1, :]
    else:
        row_coords_interp = row_coords
        grid_z_interp = grid_z

    col_ascending = col_coords[0] < col_coords[-1]
    if not col_ascending:
        col_coords_interp = col_coords[::-1]
        grid_z_interp = grid_z_interp[:, ::-1]
    else:
        col_coords_interp = col_coords

    interpolator = RegularGridInterpolator(
        (row_coords_interp, col_coords_interp), grid_z_interp,
        method='nearest', bounds_error=False, fill_value=np.nan)

    segments = stream.xy()
    swath_profiles = []

    for seg in segments:
        if len(seg) < 2:
            continue

        coords = np.array(seg, dtype=np.float64)
        pts, tangents = _resample_polyline(coords, dx)

        if len(pts) == 0:
            continue

        # Perpendicular direction: rotate tangent 90 degrees
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        n_profiles = len(pts)
        n_perp = len(perp_dists)
        Z = np.full((n_perp, n_profiles), np.nan)

        for i in range(n_profiles):
            # Sample points along perpendicular transect
            sample_x = pts[i, 0] + perp_dists * normals[i, 0]
            sample_y = pts[i, 1] + perp_dists * normals[i, 1]
            # Interpolator expects (y, x) = (row_coord, col_coord)
            query_pts = np.column_stack([sample_y, sample_x])
            Z[:, i] = interpolator(query_pts)

        swath_profiles.append(SwathProfile(
            Z=Z,
            disty=perp_dists.copy(),
            xy=pts.copy(),
            width=width
        ))

    return swath_profiles
