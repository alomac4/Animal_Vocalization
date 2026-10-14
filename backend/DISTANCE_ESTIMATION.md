# Approximate apparent-size distance, not measured depth

**Current status: all distance overlays/inference disabled.** This document
describes the older optional formula, not the active pipeline. Depth Pro has
been installed separately and preserved for future selected-frame work; both
the pinhole formula and whole-video neural depth are off by default. Settings
and generator use `distance_enabled: false`. No checkpoint is deleted.

Tracking, dog/breed models and identity association are unchanged. No neural depth
model, phone sensor readings, actual EXIF calibration or new APK is used.

The render pass calculates one estimate per accepted detector box on each frame
using original-frame coordinates. Labels show `Dog #1 | est. 3.2 m / 10.5 ft`.
Breed stays only in the statistics section. Every frame also has an uncalibrated
notice stating the assumed camera and dog size. Clipped/tiny/invalid boxes show
`distance unavailable`; fully hidden/undetected dogs receive no fabricated estimate.

## Assumptions, not an average-camera measurement

- Main-camera preset: **26mm full-frame-equivalent focal length**, a representative
  preset, not a measured average across phones. Apple lists 26mm equivalent for
  the iPhone 16 main camera; the uploaded test clip's real camera is unknown.
- Hypothetical full-frame reference long side: 36mm, mapped to video long side.
  Equivalent focal length may be specified diagonally by a device; video aspect
  cropping/stabilization changes effective FOV. This mapping is only a heuristic.
- Assumed entire projected vertical dog-box extent: **0.65m**, arbitrarily chosen
  medium-dog reference. It is NOT measured dog height, withers height, or a
  breed-derived value. Head/tail, crouching, pose and occlusion change the box.
- No focal-distance/focus-distance sensor value is used. Focal length/FOV and
  subject focus distance are different concepts. We assume focal length/FOV.
- Same preset for every dog. Do not read the number as calibrated ranging.

With W,H frame dimensions, h detected box height, F assumed equivalent focal:

```
f_pixels = max(W, H) * F_mm / 36
axial_depth_m = f_pixels * assumed_dog_extent_m / h
u = box_center_x - W/2
v = box_center_y - H/2
range_m = axial_depth_m * sqrt(1 + (u/f_pixels)^2 + (v/f_pixels)^2)
range_ft = range_m / 0.3048
```

This approximates straight-line range to the box center, not ground-plane distance
or range to the nearest surface. It assumes a frontoparallel known-size extent,
square pixels, centered principal point, no distortion or crop. Pinhole scaling
does not make the unknown camera/size assumptions correct. Upscaling is not extra
distance evidence. Changing assumed extent/focal length changes distance linearly.
Example: using 0.65m for a 0.325m projected animal would approximately double the
centered estimate. No validated accuracy or uncertainty percentage is available.
Numbers are displayed to one decimal only for UI readability, not precision.

Zoom/ultrawide lenses, portrait cropping, stabilization, sitting/running dogs,
partial occlusion, mixed dog sizes, false boxes and detector jitter can cause
large errors. A false dog box still gets a false estimate if not rejected.
This feature is illustrative; do not use it for safety or physical measurements.

## Configure on the PC

`C:\Work\Classes\Animal_Vocalization\backend\server.local.json` accepts:

```
"distance_enabled": true,
"distance_focal_mm": 26.0,
"distance_dog_extent_m": 0.65
```

Valid ranges: equivalent focal 10–200mm, extent 0.1–2m. `distance_enabled: false`
removes both range labels and notice. Worker CLI has matching options. Config
changes require safe API restart after active jobs/results finish. Existing
processed MP4s do not change retroactively. The app receives rendered video,
so its installed APK remains usable; no phone camera setting is read.

`/health` and job status include `distance_estimation` assumptions/warning.
Per-job performance reports available/unavailable range-box counts. No per-frame
distance records, annotation downloads, database or result history are added.

## Validation boundaries

Unit tests check synthetic known geometry, feet conversion, image resizing,
portrait dimensions, off-axis correction, invalid/clipped/tiny boxes, settings,
and rendering without breed/identity coupling. GPU smoke checks H.264/AAC output,
distance-bearing video path and temporary metadata. Original-video samples use
existing OC-SORT detector observations; **no real distance ground truth exists**.
Formula/integration tests are not accuracy validation. Calibration requires known
effective FOV (or f_pixels) plus dog extent/reference distance or an independent
depth measurement, ideally per dog/pose rather than one global size.

Samples: `C:\Work\Classes\Animal_Vocalization\outputs\benchmarks\distance_estimates\comparison.jpg`.
No checkpoints or tracker configuration changed.

## Primary references

- https://support.apple.com/en-us/121029 (iPhone 16 camera specification)
- https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html (pinhole camera model)