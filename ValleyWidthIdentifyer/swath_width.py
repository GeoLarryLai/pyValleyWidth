"""Extract valley width from swath profiles of a valley classification grid.

Translates the MATLAB function ``SWATHwidth.m`` from the
ElevationThresholdValleyWidth toolbox.
"""

import numpy as np
from scipy.spatial.distance import cdist


def swath_width(swath_profiles, minradius):
    """Compute valley width at each swath profile center point.

    For each cross-profile, the valley width is defined as the distance
    between the nearest out-of-valley pixels on opposite sides of the
    stream.  A local-minimum smoothing pass then replaces each width
    with the minimum width found among all points within *minradius*.

    Parameters
    ----------
    swath_profiles : list of SwathProfile
        Swath profiles from ``stream2swath()``.  Each profile's ``.Z``
        values should be from the valley classification grid (1 = valley,
        0 = hillslope, 2 = stream).
    minradius : float
        Search radius (map units) for the moving-minimum smoothing.

    Returns
    -------
    np.ndarray
        (N x 4) array where columns are:
        0 - x coordinate of profile center,
        1 - y coordinate of profile center,
        2 - raw valley width,
        3 - minimum width within *minradius*.
    """
    all_stream_widths = []

    for swath in swath_profiles:
        Z = swath.Z           # (n_perp, n_profiles)
        disty = swath.disty   # (n_perp,)
        xy = swath.xy         # (n_profiles, 2)
        total_width = swath.width
        half_width = total_width / 2.0
        n_profiles = xy.shape[0]

        widths_this_stream = np.zeros((n_profiles, 4))

        for pp in range(n_profiles):
            profile_vals = Z[:, pp]

            # Out-of-valley: classification < 1 (i.e. 0 = hillslope)
            out_of_valley = profile_vals < 1.0
            out_dists = disty[out_of_valley]

            pos_dists = out_dists[out_dists > 0]
            neg_dists = out_dists[out_dists < 0]

            # If profile never leaves valley on one side, cap at half_width
            if len(pos_dists) == 0:
                min_pos = half_width
            else:
                min_pos = np.min(pos_dists)

            if len(neg_dists) == 0:
                min_neg = half_width
            else:
                min_neg = np.min(np.abs(neg_dists))

            valley_w = min_pos + min_neg

            widths_this_stream[pp, 0] = xy[pp, 0]
            widths_this_stream[pp, 1] = xy[pp, 1]
            widths_this_stream[pp, 2] = valley_w
            widths_this_stream[pp, 3] = 0.0  # placeholder

        # Moving-minimum smoothing per stream
        if n_profiles > 0:
            dists = cdist(widths_this_stream[:, :2],
                          widths_this_stream[:, :2])
            in_radius = dists < minradius
            for ppp in range(n_profiles):
                nearby_widths = widths_this_stream[in_radius[:, ppp], 2]
                widths_this_stream[ppp, 3] = np.min(nearby_widths)

        all_stream_widths.append(widths_this_stream)

    if len(all_stream_widths) == 0:
        return np.empty((0, 4))

    return np.vstack(all_stream_widths)
