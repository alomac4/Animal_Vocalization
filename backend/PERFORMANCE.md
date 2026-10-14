# Latest: GPU crop assembly instead of host tile assembly

## Identity fragmentation investigation (not solved)

### Original-video sequential comparison

The original 4,725-frame, 1280x720 video is now available in `test_video` and was
processed with identical cached dog detections. Final identities after the same
presence filter: current 31, OC-SORT 28, BoT-SORT without neural ReID 31,
FastTracker 32. OC-SORT is selected as the sequential motion tracker, but does
**not** solve the reported five-dog clip. It is observation-centric Kalman
tracking, not a newly trained RNN or a learned animal identity model.
Breed remains outside all association decisions. Comparison videos, contact
sheets and full report are in `outputs\benchmarks\original_video_tracking`.
Model/SR precision and video-frame coverage remain unchanged. No APK update.

The retained real-video job reports 4,725 frames / 157.6575 seconds and **75
confirmed identities**, zero linked fragments. Its stage timings finally identify
enhancement as the largest contributor: dog 80.7633 s, neural SR 241.8333 s,
breed 113.4825 s, total 458.9447 s. The SR model made 48,829 calls on 18,792
crops. These are actual job metrics, not earlier synthetic timing projections.

Revised tracking has an original-crop appearance gallery, 15-second dormant
recovery, mutual-best margin-separated final linking, transitive lifetime checks,
and post-link duration filtering. Breeds never enter these decisions. Synthetic
tests cover five IDs with changing breeds and four-second missing detections.

Dog-only diagnostic on the retained **already rendered** MP4: 105 candidates,
2 online recoveries, 65 unfiltered identities, 34 rejected brief identities,
**31 final identities**. The user reports five actual dogs, so this remains
inadequate. Burned-in overlays, downscaling and re-encoding also mean this is not
an exact original-clip comparison. It is not evidence of identity correctness,
and the threshold was not tuned to force five dogs or equal observation counts.
Original-video testing and stronger animal re-identification are still needed.

The minimum for this 4,725-frame clip is 142 actual observed frames (max of
3 seconds, 3% of clip frames, 5% of longest observed identity). A 100-frame
identity is rejected only after link pooling. Legitimate brief visitors can be
lost; long-lasting false positives can survive. Counts are tracking hypotheses,
not verified unique animals. Per-identity observed-frame counts are now included.
Report: `C:\Work\Classes\Animal_Vocalization\backend\runtime\identity-recovery-video.json`.

## GPU crop assembly measurements

Real-ESRGAN now quantizes only useful tile cores and assembles the native 4x
crop on CUDA. It copies one BGR uint8 image per enhanced crop to the CPU,
instead of repeatedly transferring full tiles including unused borders.
OpenCV area resize, model weights, precision, detection sizes, frame coverage,
tracking and final non-unknown breed voting are unchanged.

Seven warmed/interleaved trials on identical inputs, same model and batches:

| Input (height × width) | Previous median | GPU assembly median |
| --- | ---: | ---: |
| 65 × 97 | 2.554 ms | 2.500 ms |
| 215 × 315 | 16.420 ms | 12.255 ms |
| Four 217 × 247 crops | 41.671 ms | 29.662 ms |
| Four varied crops | 107.938 ms | 76.681 ms |
| 600 × 900 | 93.329 ms | 66.738 ms |

All six cases (including bypass) matched the reference **pixel-for-pixel**.
Assembly unit tests also cover BGR ordering, tile edges, bypass, area resize,
empty input and bounds. Keeping complete uint8 crop buffers on-device uses
slightly more bounded VRAM but transfers fewer bytes. Filling VRAM is not a goal.

End-to-end comparison: 120 frames at 25 fps, generated from sampled gate and
backyard stills. Two trials with reversed variant order, includes initialization,
decode, tracking, final rendering and audio mux; excludes network:

| Assembly | First trial | Second trial | Upscale stage (second trial) |
| --- | ---: | ---: | ---: |
| Host tiles | 8.3917 s | 6.3790 s | 2.4774 s |
| GPU crop | 5.9475 s | 5.9868 s | 1.8747 s |

The first host run has startup effects. The warmed full-pipeline improvement is
approximately **6–7%**, not the misleading 29% implied by comparing only first
trials. Enhancement itself was about 24% faster in the warmed video trial.
Output frames, detected crops and consensus counts were identical. Transfers
fell from 603 to 234; output host bytes from 806,823,600 to 541,462,992.
These measurements do not predict an eight-minute user-video runtime.

Report: `C:\Work\Classes\Animal_Vocalization\backend\runtime\upscaler-transfer-profile.json`.
Reproduce with project Python and
`C:\Work\Classes\Animal_Vocalization\scripts\profile_upscaler_transfers.py --video`.
The next real-video job's `performance` object is needed to identify its bottleneck
before expiry. This diagnostic remains temporary: no stored results or database.
Backend-only change; existing APK/URL/token remain usable.

## Previous: bounded batching, decode prefetch and identity-level consensus

The current defaults are dog batch **1**, breed crop batch **4**, and equal-sized
Real-ESRGAN tile batch **8**. Decoding overlaps inference via a two-batch CPU queue.
Larger frame batches remain configurable but are not the default because they
did not produce a dependable speed improvement on this RTX 4090.

Warmed model/crop-only test: 24 benchmark stills, with repeated trials:

| Dog / crop / tile batch | Median FPS |
| --- | ---: |
| 1 / 4 / 4 | 24.48 |
| 1 / 4 / 8 | 24.37 |
| 1 / 8 / 8 | 24.25 |
| 2 / 8 / 8 | 23.99 |
| 4 / 8 / 8 | 23.78 |
| 8 / 8 / 8 | 22.78 |

End-to-end local synthetic video: 120 frames (4.8 seconds), made from 12 sampled
gate stills repeated ten times each. Two trials, reversed order; includes model
initialization, decode, tracking, consensus, render and mux, excludes network:

| Configuration | Trial 1 | Trial 2 |
| --- | ---: | ---: |
| Serial decode, 1 / 4 / 4 | 6.1724 s | 4.2627 s |
| Decode prefetch, 1 / 4 / 8 | 4.2669 s | 4.3749 s |
| Decode prefetch, 4 / 8 / 8 | 4.4487 s | 4.1793 s |

The first serial run includes startup effects; comparing it alone to prefetch
would exaggerate gains. The remaining runs are about 4.2–4.4 seconds. No material
end-to-end speedup is established. Frame4 batching reduced SR model calls from
290 to 128 but did not reliably reduce total runtime. Batched versus serial SR
differed by at most one uint8 pixel level on three test inputs; this is not proof
of equivalent breed accuracy. Reports: `backend\runtime\batched-pipeline-profile.json`
and `backend\runtime\prefetch-video-profile.json`. These are not the user's clip.

Final consensus now links only unambiguous nearby/similar-color nonoverlapping
track fragments within six seconds, pools their non-unknown votes, and applies
one final breed to every rendered observation in the identity. It does not use
breed for matching. IDs and labels remain estimates, not guaranteed animal ReID.
Per-job diagnostics include confirmed/linked tracklet counts and effective batch
sizes. No APK update is required. Historical measurements below describe earlier
versions, not the active pipeline.

## Neural upscaler validation (historical)

Real-ESRGAN general-x4v3 CUDA FP16 is now available for crop enhancement.
Same-crop warmed timings on four actual dog crops (three repeats per method):

| Crop dimensions | Lanczos | Neural SR |
| --- | ---: | ---: |
| 315 × 215 | 3.714 ms | 17.665 ms |
| 241 × 270 | 4.271 ms | 16.691 ms |
| 205 × 179 | 2.888 ms | 7.141 ms |
| 247 × 217 | 4.301 ms | 18.466 ms |

These measure only crop enhancement, excluding loading, dog/breed inference,
decode, render and network time. All samples come from the same gate sequence;
they do not represent every crop size or scene. Sequential tiled enhancement
adds work per dog per frame; no full-video runtime improvement is claimed.

Tiled versus full SR matched exactly on the tested 97 × 65 random input.
The five-frame CUDA integration test enhanced five crops over 30 tiles and
passed tracking consensus, H.264/AAC, download and deletion. Total smoke time
was 2.2871 seconds including initialization; it is not a steady-state FPS result.

The four crop comparisons changed all four eligible breed outputs: two changed
to “malamute”, one from “German shepherd” to “chinese rural dog”, and one from
“German shepherd” to unknown. This is evidence of a preprocessing effect, not
evidence of an accuracy improvement. Exact images, labels and scores are at
`C:\Work\Classes\Animal_Vocalization\outputs\benchmarks\neural_crop_upscaling\report.json`
and `comparison.jpg` in the same folder. No breed ground truth was used.

# Current 1536-resolution pipeline

The dog model runs on original decoded frames before output downsampling, with
1536 × 1536 aspect-preserving letterboxing (`rect=False`). Smaller images are
enlarged by model preprocessing. Breed model input remains 960 × 960. Clean
dog-box crops are enlarged using OpenCV Lanczos4 interpolation toward a 960-pixel
long side, capped at 4× enlargement. This is not neural super-resolution or a
segmentation mask. Source detail is not recovered by interpolation.

The installed Ultralytics version uses `quantize=16` for FP16 and `quantize=32`
for FP32. Configuration keeps a `half` boolean for compatibility. Breed crops
are batched in groups of up to four, so dense scenes do not allocate an unbounded
number of enlarged crops at once. Every frame is processed; tracking is now added
without skipping dog or breed inference. Final breed voting uses a second render
pass; the measurements below predate tracking and do not measure this new version.
The video still has a maximum 1280-pixel long side, independent of model input.
Original dog coordinates are rescaled for the rendered video. Models and
training checkpoints are unchanged. Higher resolution does not prove accuracy.

## Measurements on this PC

Report: `C:\Work\Classes\Animal_Vocalization\backend\runtime\pipeline-profile.json`.
24 sampled benchmark stills repeated five times (120 processing iterations):

| Controlled configuration | FPS | Time for 120 frames |
| --- | ---: | ---: |
| Square 960 / FP32 / serial crops / no explicit crop enlargement | 28.42 | 4.22 s |
| Square 1536 / FP32 / serial crops / Lanczos enlargement | 15.79 | 7.60 s |
| Square 1536 / FP16 / batch up to 4 / Lanczos enlargement | 25.35 | 4.73 s |

FP16/batching is about 1.61× faster than unoptimized 1536 here, but still slower
than the controlled 960 pipeline. Actual FP16 and square input sizes were checked
on the predictors. This is not the user's exact video, not an accuracy test, and
not an end-to-end runtime estimate. Initialization, decode, encode and network
are excluded. The 960 variant is not an exact replay of the old production
worker, which downsampled input and could use different letterbox dimensions.

Reproduce:

```powershell
Set-Location 'C:\Work\Classes\Animal_Vocalization'
& '.\.venv\Scripts\python.exe' '.\scripts\profile_video_pipeline.py'
```

## Per-job timings

Job status now includes an in-memory `performance` object with dog, crop-upscale,
breed, decode, drawing/encoder-pipe times, total seconds, video duration and frames.
Total includes loading, encoding drain and muxing; stage totals do not include
those or first-frame decode. The encoder-pipe stage includes pipe waits rather
than full asynchronous encoder time. Network upload/phone download are excluded.
GPU stage time includes postprocessing and host result transfer. Metrics expire
with the job, with no database/permanent app inference results.

All 19 backend tests passed. Real CUDA smoke passed with five decoded frames,
five dog boxes/crops/eligible breed labels, H.264 video, AAC audio, download and
deletion. Log: `backend\runtime\1536-validation.log`. Historical README test
sections describe old configurations. The live server has been restarted with
1536 dog input, 960 breed input, FP16, batch four, and Lanczos4.

## Further optimization options (not yet implemented)

1. Tracking and final clip-wide majority labels are now implemented, but breed
   predictions are still evaluated on every detected crop. Periodic crop inference
   could reduce GPU time; it is not enabled because it would change which frames
   contribute votes and requires separate association/quality evaluation.
2. Profile the actual video with the new timings before choosing the bottleneck.
3. Export/check TensorRT FP16 engines, validating outputs against these models.
4. If encoding is significant, test FFmpeg H.264 NVENC rather than CPU libx264.
   This FFmpeg build lists NVENC, but its runtime/quality has not been tested.
5. Keep loaded model workers alive between videos to reduce startup latency.

These backend-only changes require no new APK, URL, token or phone reinstall.

## Track-majority rendering

Original detector boxes are assigned video-local IDs by constant-velocity,
IoU-gated Hungarian matching, with roughly one second of lost-track retention.
Final breed labels use the mode of non-unknown predictions per track; ties use
summed detector scores and alphabetical order. All observations of the track use
that label in the second decode/render pass. All-unknown tracks stay unknown.

New per-job timings include `tracking_seconds` and `render_decode_seconds`.
No decoded frame cache or permanent inference record is created: only temporary
boxes/ID records are spooled within the job directory and removed. Model GPU
work remains unchanged; second-pass decode/spooling adds overhead. No runtime
improvement is promised by this change. Crowded scenes, camera motion, and
occlusions can cause ID switches/fragments and therefore incorrect vote grouping.