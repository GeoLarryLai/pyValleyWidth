# pyValleyWidth

Python tools to measure **river valley width** from a digital terrain model (DTM), built on **[pytopotoolbox](https://github.com/TopoToolbox/pytopotoolbox)** (TopoToolbox for Python).

These Python codes are inspired by Paul Morgan’s **[ElevationThresholdValleyWidth](https://github.com/PMonroeMorgan/ElevationThresholdValleyWidth)** workflow originally written in MATLAB for **[TopoToolbox 2](https://github.com/wschwanghart/topotoolbox)**.

> [!NOTE]
> If you use **pyValleyWidth** in your work, please cite Paul Morgan’s paper (about to be published): Morgan, P. M., Grant, A., Struble, W., LaHusen, S., and Duvall, A.: *The damability function: A probabilistic approach to regional landslide dam susceptibility analysis applied to the Oregon Coast Range, USA*, EGUsphere [preprint], [https://doi.org/10.5194/egusphere-2025-580](https://doi.org/10.5194/egusphere-2025-580), 2025. Also cite **TopoToolbox** / **pytopotoolbox** as appropriate for your work.

> [!WARNING]
> **pyValleyWidth** is currently under active development and has not yet been peer-reviewed and fully validated. Please use with caution.

---

## What you get in this repository

| Item | Purpose |
|------|---------|
| **`ValleyWidthIdentifyer/`** | Importable package: valley classification, swath widths, optional variable rim thresholds, stream-network smoothing, \(k_{\mathrm{sn}}\), and river-profile figures. |
| **`valley_width_example.ipynb`** | End-to-end example on the **Big Tujunga** DEM bundled with topotoolbox: flow routing, optional largest catchment mask, valley width (two modes), maps, \(k_{\mathrm{sn}}\), longitudinal / χ–Z profiles, and threshold / excess topography. |
| **`Method_overview.png`** | Visual summary of the valley-width method (from Morgan et al., in review). |

The notebook is the main **user guide**: it walks through each step and includes **parameter notes** for `dem2widths` (stream area, HAND threshold vs. variable rim mode, swath spacing, smoothing, saturated transects, etc.) and for the longitudinal **valley-width twin-axis** plots.

---

## Method at a glance

Valley width is derived from height above the channel network, valley polygons, and cross-stream swath profiles along the river. Perpendicular profiles can overestimate width on tight bends; the workflow can resample / smooth widths along the network (moving window) so the final map is more stable—see the figure below. Some half-backed functionalities available to automated the maximum valley width/depth, but they are unstable and under active development.

![Method overview: hillshade and river path, relative elevation, profile-based widths, resampled valley width](./Method_overview.png)

*Figure after Morgan et al. (in review); panels illustrate relative elevation, perpendicular profiles (with bend artefacts), and resampled valley width.*

---

## Pipeline (high level)

1. **Terrain and streams** — Flow routing and stream extraction with topotoolbox; optional **largest connected catchment** so all later analyses use one main basin (`StreamObject.klargestconncomps(1)`-style workflow).
2. **Valley definition** — Two options in **`dem2widths`** (see notebook §3):
   - **Constant HAND threshold** (`max_valley_width=False`): pixels below a fixed height above the nearest stream are “valley”; returns `(allwidths, DV)`.
   - **Variable rim along the channel** (`max_valley_width=True`): per-transect ridge detection, rim height as a fraction of the dominant peak HAND, smoothing along the network, then a compatible `DV`; returns `(allwidths, DV, allvalleydepth)`.
3. **Widths from swaths** — Cross-stream profiles on the classified valley raster; optional **saturation** filtering when transects are too short.
4. **Maps on the network** — Raw widths at sample points; **gap-filled, vertex-averaged, Gaussian-smoothed** width fields on the stream network for cleaner maps (see `smooth_valley_width_on_stream_network` and related helpers).
5. **Geomorphology add-ons (example notebook)** — \(k_{\mathrm{sn}}\) from slope–area regression with fixed concavity, longitudinal and χ–Z profiles (trunk highlighted), twin-axis trunk profiles (width ± valley top in variable mode), and **threshold / excess topography** at a chosen slope (e.g. 35°).

---

## Installation and quick start

1. Create a conda (or venv) environment with **`topotoolbox`** installed (same environment as your Jupyter kernel).
2. Place the ValleyWidthIdendifyer package folder under your working project directory.
3. Open **`valley_width_example.ipynb`** and run top to bottom.
4. For your own DTM, copy the notebook sections and adjust **`streamarea`**, **`elevthreshold`** or **`max_valley_width`** settings, **`swath_dx`**, **`minradius`**, and swath length / `'auto'` options as documented in the notebook.

Import in your own scripts:

```python
from ValleyWidthIdentifyer import dem2widths
# other symbols: see ValleyWidthIdentifyer/__init__.py
```

---

## Package layout (`ValleyWidthIdentifyer`)

- **`dem2widths`** — Orchestrates routing, short-stream removal, valley classification, swaths, width extraction, and drainage area / gradient at width samples.
- **`valleyclass_elev`**, **`swath_width`**, **`peak_valley_width`** — Valley mask construction and rim / peak logic for the variable-threshold mode.
- **`stream_utils`** — Short-stream removal, stream-to-swath geometry, `SwathProfile`.
- **`valley_width_stream`** — Flatten stream coordinates, smooth width on the network, trunk sampling order.
- **`ksn_analysis`** — Slope–area diagnostics, \(k_{\mathrm{sn}}\), smoothing and NaN fill along streams.
- **`river_profile_plots`** — Longitudinal, χ–Z, \(k_{\mathrm{sn}}\) / valley-width twin panels, threshold and excess topography figures.
