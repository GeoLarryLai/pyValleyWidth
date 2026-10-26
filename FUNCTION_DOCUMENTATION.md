# pyValleyWidth function documentation

Public functions exported by `pyValleyWidth` (version 0.1.1). Signatures and summaries are taken from the docstrings; see each docstring for full parameter details.


## `dem2widths`

### `dem2widths(dem, streamarea, elevthreshold=None, swath_dx=None, minradius=None, swath_width_param='auto', units='pixels', plot=False, klargest_conncomps=None, main_trunk_only=False, nodata_value='auto', swath_width_quantile=0.99, swath_width_safety=1.5, max_valley_width=False, n_thr=300, dw_smooth=5, min_dwdt=0.0, profile_smooth_radius=0.0, dv_smooth_radius=None, max_dwdt=None, clip_to_divides=True, n_jobs=-1)`

Full pipeline from DEM to valley-width measurements.


## `valleyclass_elev`

### `valleyclass_elev(dem, stream, flow, elevthreshold, plot=False, nodata_mask=None)`

Classify each DEM pixel as valley (1), stream (2), or hillslope (0).


## `swath_width`

### `swath_width(swath_profiles, minradius)`

Compute valley width at each swath profile center point.


## `peak_valley_width`

### `per_transect_edge_thresholds(swath_profiles_dz, cellsize=None, n_thr=300, dw_smooth=5, min_dwdt=0.0, profile_smooth_radius=0.0, max_dwdt=None, clip_to_divides=True, n_jobs=-1)`

Per-transect spillover-based valley-depth threshold.

### `build_dv_from_thresholds(dem, stream, DZ, profile_xy, thresholds, nodata_mask=None, max_dist=None, smooth_radius=0.0)`

Build a valley-classification grid from a variable along-channel threshold.


## `stream_utils`

### `removeshortstreams(stream, minlength)`

Remove stream segments shorter than *minlength* (map units).

### `stream2swath(stream, grid, dx, width)`

Generate cross-stream swath profiles for a grid along a stream network.

### `SwathProfile(Z: numpy.ndarray, disty: numpy.ndarray, xy: numpy.ndarray, width: float) -> None`

Container for swath profile data along a single stream segment.


## `ksn_analysis`

### `slopearea(stream_obj, dem, acc_grid, plot_fit=True, savefig=None)`

Bin stream nodes in log(S)–log(A) space, linear fit, return θ and k_s.

### `calculate_ksn(stream_obj, dem, acc_grid, theta)`

Channel steepness index k_sn = S / A^θ along each stream segment.

### `smooth_stream_values(stream_obj, values, window_size=20)`

Gaussian smooth *values* independently along each ``stream_obj.xy()`` segment.

### `fill_nan_stream_segments(stream_obj, values)`

Linearly interpolate NaNs along each stream segment (xy walk order).


## `valley_width_stream`

### `stream_xy_flat_coords(stream_obj)`

Flatten ``stream_obj.xy()`` map coordinates (same order as fill/smooth).

### `smooth_valley_width_on_stream_network(allwidths, stream_obj, max_match_dist=150.0, window_size=100, width_column=2)`

Snap width samples to nearest stream vertices, mean, fill NaNs, smooth along network.

### `valley_width_smoothed_on_stream(allwidths, stream_obj, max_match_dist=150.0, window_size=100, width_column=2)`

Same pipeline as the smoothed-width map: return flat coords + smoothed vertex field.

### `trunk_valley_width_from_s2_field(allwidths, s2_stream, st_trunk, max_match_dist=150.0, window_size=100, s2_field=None, width_column=2)`

Smoothed widths on *s2_stream*, sampled onto *st_trunk* NAL nodes, gap-filled along *st_trunk* only.

### `xy_flat_nal_walk_order(stream_obj)`

NAL index at each ``xy()`` visit (same walk as ``fill_nan_stream_segments``).

### `nal_from_xyflat_last(stream_obj, values_flat, nal_len=None)`

Collapse xy-flatten vector to NAL; last visit wins for duplicate nodes.

### `xyflat_from_nal_expand(stream_obj, values_nal)`

Expand NAL vector to xy-flatten length (one value per ``xy()`` visit).


## `river_profile_plots`

### `longest_tributary_subgraph(stream_obj)`

Return a ``StreamObject`` containing only the longest DFS segment (main channel proxy).

### `plot_stream_ksn(stream_obj, dem, ksn_values, title='Normalized Channel Steepness (ksn)', vmin=0, vmax=None, figsize=(15, 10), savefig=None)`

Hillshade + stream nodes colored by *ksn_values* (same walk order as ``calculate_ksn``).

### `plot_longitudinal_trunk_colored_ksn(S_all, st_trunk, dem, ksn_st_smoothed, vmin=0, vmax=None, theta=None, figsize=(10, 8), savefig=None)`

Gray branches as in χ–Z: point per branch colored node, trunk colored by k_sn by distance.

### `plot_longitudinal_trunk_ksn_valley_width_twin(s2, st, dem, dem_imposed, acc, allwidths, theta=-0.45, ksn_window=200, valleywidth_window=100, max_match_dist=150.0, vw_outline_sigma_pts=4.0, ksn_vmin=0, ksn_vmax=None, ksn_clim=None, valleywidth_ylim=None, figsize=(16, 6), savefig=None, allvalleydepth=None, valleydepth_window=None, ma_window_m=500.0)`

Long profile (distance-Z): gray s2 scatter, trunk colored by k_sn, twin = valley width.

### `plot_chi_z_trunk_colored_gray_network(S_all, st_trunk, dem, acc, ksn_st_smoothed, vmin=0, vmax=None, theta=None, figsize=(10, 8), savefig=None)`

All streams as light gray χ–Z points; trunk colored by smoothed k_sn.

### `plot_chi_z_trunk_ksn_valley_width_twin(st, dem, acc, dem_imposed, allwidths, theta=-0.45, ksn_window=100, vw_window=100, max_match_dist=150.0, ksn_vmin=0, ksn_vmax=None, valleywidth_ylim=None, figsize=(16, 6), savefig=None)`

χ–Z scatter colored by smoothed k_sn + twin axis stream-smoothed valley width vs χ.

### `plot_threshold_and_excess_topography(dem, slope_degrees=35.0, figsize=(15, 12), savefig=None, basin_mask=None)`

Threshold (``fsm2d``) topography and DEM minus threshold surface for one slope angle.


## `transect_diagnostic_plot`

### `plot_transect_diagnostic(dem, fd, stream, xy, *, min_dwdt=0.0, max_dwdt=None, clip_to_divides=True, swath_width='auto', swath_dx=None, hand=None, title=None, color='#377eb8', figsize=(16, 6), save_fig=False, filename=None, dpi=150, savefig=None)`

Three-panel diagnostic figure for one cross-section.
