# BARK PC inference server

## Result presentation (app 1.0.2)

### Depth disabled; model retained for selected-frame experiments

Distance/depth inference is disabled in the local config, settings defaults,
worker CLI default and configuration generator. New jobs after API settings
reload show only dog IDs and boxes; breeds stay in the statistics section.
No Depth Pro model process or distance notice is started when disabled.
Tracking, detector, breed and neural crop enhancement are unchanged.

The official Depth Pro checkpoint is preserved at
`C:\Work\Classes\Animal_Vocalization\backend\assets\depth_pro.pt`, alongside
its manifest; source/licence and `backend\.depth-venv` are retained. Selected
original-frame experiments can use `scripts\validate_depth_pro.py`, which
processes four samples rather than all video frames. Do not enable whole-video
depth just to use the stored model on selected frames. It remains experimental:
no measured ground-truth distances or validated accuracy. No new APK is needed.
Available downloads are preserved until a safe idle restart. Existing rendered
distance text cannot be removed without reprocessing the video.

### Identity recovery and presence filtering

The configured `tracker_backend` is now `ocsort`: installed Ultralytics OC-SORT
with observation-centric Kalman updates, direction history, low-confidence
association enabled, 15-second lost-track buffer, and no neural ReID model.

An optical-flow gap-location experiment exists in `gap_motion.py` and the
development adapter's `ocsort_flow` variant. It is intentionally **not** in the
server configuration's allowed backends. Same original clip: 30 final IDs with
flow vs 28 without; association/decode 60.672 s vs 13.617 s. Keep it diagnostic,
not active. It follows visible points for at most 0.75 seconds since a confirmed
detection and uses estimates only as association hints: no rendered imaginary
boxes, observed-frame increments, crop enhancement or breed votes. Full report
and preview: `outputs\benchmarks\gap_motion_tracking\index.html`.
`motion_color`, `botsort`, and `fasttrack` remain selectable alternatives.
Sequential association runs on every frame in order, including empty detections;
only original observed detector boxes are rendered. Shared final breed voting,
confirmation, fragment linking and presence filtering remain enabled. The `lap`
assignment dependency is installed and pinned in project requirements.

Original-video test (4,725 frames, same cached detections): current 31 final IDs,
OC-SORT 28, BoT-SORT without neural ReID 31, FastTracker 32. **This does not solve
the user's five-dog clip.** Reduced counts do not establish correct identities.
Comparison videos/report: `outputs\benchmarks\original_video_tracking\index.html`.
No count cap, equal-duration enforcement or breed-based ID association is used.

Breed predictions are used **only for final voting**, never for association,
track birth, recovery, or linking. The tracker retains up to 12 original-crop
HSV appearance views per track and tries mutually-best, margin-separated
recovery before starting a new ID. Dormant identities may reconnect within
15 seconds, using bounded position/area plus appearance. Post-pass links use
the same evidence and explicitly prohibit overlapping lifetimes, including
transitive links. This is heuristic color/motion matching, not learned animal
re-identification. Similar dogs, camera motion, long exits and pose changes can
still split or swap identities. It does not force any fixed number of dogs.

After linking, reject identities with fewer actual observed frames than the
maximum of: three seconds of detections, 3% of decoded clip frames, and 5% of
the longest-observed identity. The three-second requirement is capped by clip
length. Defaults are configurable via `track_presence_seconds`,
`track_clip_presence_ratio`, and `track_presence_ratio`; set ratios to zero
to disable the corresponding tests. These are duration heuristics, **not proof
of a false detection**: they intentionally risk suppressing genuine brief visitors.
They do not equalize dog durations and do not reject a dog just for appearing
longer than another. On a 4,725-frame ~30fps clip the clip-relative floor is
142 observed frames, so a 100-frame isolated identity is suppressed; fragments
are pooled first so short pieces of a long identity need not be discarded.

`dog_statistics.observed_frames` reports the pooled actual detections, not
first-to-last frame span. Per-job diagnostics include `recovered_track_count`,
`presence_rejected_identities`, `presence_rejected_tracklets`, and
`presence_minimum_frames`. Existing APK accepts the extra metadata without an
update. Existing rendered videos cannot be corrected without reprocessing.

Video overlays now contain only `Dog #<id>` and boxes, not breed names/scores.
Final identity-level breed voting is unchanged. The authenticated job response
includes `dog_statistics`: one row per confirmed/linked identity with `dog_id`,
`name`, `breed` (nullable), `votes`, and `known_votes`. IDs match the video's
consecutive display IDs exactly. Vote counts are not calibrated confidence.
The Android app displays these under the result video. Metadata is in memory
only and expires/deletes with the job; no database or annotation file is added.
Older completed videos are unchanged; reprocess to use dog-ID-only overlays.
`/health` reports `video_labels: dog_id_only`. Requires the updated Android APK
for the statistics section; server URL/token do not change.

## Neural crop upscaling (Real-ESRGAN)

Before breed inference, smaller dog crops are now enhanced by the official
`realesr-general-x4v3` pretrained model on CUDA, using the same FP16 setting as
the detectors. Original full frames still go to the dog detector; original crop
colors still go to tracking. Only the breed detector sees enhanced crops.
The downloaded video is not super-resolved and still has its existing size cap.

Setup (no new pip dependencies or training-environment changes):

```powershell
& 'C:\Work\Classes\Animal_Vocalization\backend\.venv\Scripts\python.exe' 'C:\Work\Classes\Animal_Vocalization\backend\install_upscaler.py'
```

Weights are stored in `backend\assets\realesr-general-x4v3.pth`; their official
release URL, SHA256, architecture revision and license are recorded in
`backend\assets\upscaler-manifest.json`. The worker verifies that checksum and
loads the state dict with `weights_only=True`. The locally adapted compact
architecture uses existing PyTorch, not an incompatible BasicSR install. The
upstream BSD-3-Clause license is in `backend\third_party\Real-ESRGAN-LICENSE.txt`.

`upscaler_weights` in the private server config enables neural mode. Removing
that setting selects the previous Lanczos-only mode explicitly. No silent CPU
or interpolation fallback occurs if a configured neural model fails.

Native model output is 4x. It is downsampled, when needed, to the existing
aspect-preserving crop target: 960-pixel long side, with at most 4x enlargement.
Crops already at/above that target bypass enhancement. Tiling uses 192-pixel
cores with 36-pixel contextual borders, limiting GPU activation memory. Equal-sized
tiles are batched across crops (up to eight tiles per model call). Only useful
tile cores are quantized and assembled on the GPU; contextual borders are not
copied to the CPU. One host transfer per enhanced crop replaces per-tile-batch
transfers. Final OpenCV area resizing stays unchanged. Per-job diagnostics expose
`upscaler_host_transfers` and `upscaler_host_bytes`; `/health` identifies this
implementation as `upscaler_assembly: gpu_core_stitch_v1`. Models load once
per video worker, not once per crop. Smaller output factors do not save the native
4x model's work. Even tiled processing adds latency; this is not a speed fix.

`performance.upscale_seconds` includes enhancement. Neural/bypassed crop and tile
counts are reported per job. Super-resolution can invent detail and change breed
predictions; it has not been demonstrated to improve breed accuracy. It cannot
fix false-positive dog boxes or tracking identities. No new APK is required.

GPU comparison/check script:
`C:\Work\Classes\Animal_Vocalization\scripts\validate_neural_upscaler.py`.
Its local outputs are in `outputs\benchmarks\neural_crop_upscaling`.

Authenticated FastAPI service. No database, accounts, result history, JSON
annotation download, or MCP server. It burns dog boxes into a playable H.264 MP4
and preserves the first input audio stream as AAC when available.

## Activation and measured comparison

The recorded four-crop check measured approximately 7–18 ms per neural crop,
versus 3–4 ms for Lanczos, excluding loading/warmup. Breed predictions changed
and did not consistently improve; no breed accuracy improvement is established.

Activation requires restarting the API after changing its configuration. A running
API retains its loaded settings: adding weights to disk/config does not hot-reload
the job manager. Check authenticated `/health` for `crop_upscaler` before assuming
neural mode is active. The idle-restart script waits for existing temporary jobs
and downloads to be released rather than erasing available results early.

## Two-stage inference

Phone install/connect instructions are also in
`C:\Work\Classes\Animal_Vocalization\CONNECT_PHONE.md`.

1. The trained dog-only detector finds dogs in each original decoded frame on
   `cuda:0`, using **1536 × 1536**, aspect-ratio-preserving letterboxing and
   `rect=False`. Low-resolution frames are resized up by model preprocessing;
   high-resolution frames are scaled to fit. Nothing is stretched to a square.
2. Each valid dog box is clipped to the frame and copied as an unannotated crop.
3. With `upscaler_weights` configured, Real-ESRGAN enhances small dog crops on
   CUDA toward a 960-pixel long side, with an at-most-4× enlargement limit and
   no shrinking of larger crops. Without that setting, the explicit alternative
   is OpenCV Lanczos4 interpolation. These are boxed crops, not segmentation masks;
   enhanced detail is not verified recovered detail.
   The breed model letterboxes these crops to 960 × 960 on the same GPU. Small
   crops capped at 4× may still need model-side resizing. Crops are processed in
   bounded batches of up to four; FP16 is enabled for both models.
   It never searches the full frame for additional boxes.
4. An eligible crop breed must score at least 0.25, include the crop center,
    and cover at least 20% of the crop. The tracker associates original dog boxes
    using global one-to-one Hungarian matching, bounded center motion and a small
    central HSV color histogram (not neural re-identification). Confirmed tracks
    match strong detections first; detections scoring 0.25–0.50 may continue a
    confirmed track but never create one. A new candidate needs a score of at
    least 0.60, then three observations scoring at least 0.50 within one second
    to be confirmed. Unconfirmed candidates disappear after a half-second gap;
    confirmed lost tracks are kept for three seconds. Breed labels never affect
    identity association. Only confirmed tracks are rendered, including their
    initial detections retrospectively; unconfirmed boxes are dropped from the
    entire output. Display IDs are consecutive across confirmed tracks, not
    candidate numbers. IDs reset for each video. No boxes are invented when
   detection is missing; detection and breed inference still run on every frame.
5. At the end of inference, short track breaks may be linked if nearby position,
   size and original-crop color agree, and each side has only one eligible match.
   The linking window is six seconds. Overlapping lifetimes, missing appearance,
   or ambiguous alternatives do not link. Breed predictions never decide identity.
   Votes are pooled across linked track fragments, then each resulting identity
   picks its **most frequent non-unknown
   breed**. Every eligible crop prediction gets one vote. Unknown/invalid results
   are excluded. Ties use summed detector scores, then alphabetical label order.
   This is a plurality (mode), not necessarily over 50% of all known votes.
   An identity without known votes remains unknown. Even one known vote can win.
6. The input is decoded again and rendered using the original boxes plus
   `Dog #<id>` for every accepted observation of that confirmed track. Final
   voted breeds are returned separately in `dog_statistics`, matched by the
   same display IDs. Breed text is not burned into the MP4; the client never
   draws video overlays.

## Bounded batching and decode overlap

The decoder prefetches into a two-batch queue while inference runs. Frame batches
are bounded by a 128 MiB decoded-frame budget; every frame is still processed in
order. Configurable `dog_batch` supports 1–8, `breed_batch` 1–8, and
`upscaler_batch` 1–8 equal-sized tiles. No concurrent calls share a YOLO predictor.
The measured default is dog batch 1, crop batch 4 and tile batch 8: larger frame
batches did not reliably improve the tested workloads. Spare VRAM is not evidence
of spare compute. See `backend\PERFORMANCE.md` for timings and limitations.

Final label stability is guaranteed per resulting identity, not per physical dog:
unlinked fragments can still receive different labels and erroneous associations
can pool votes incorrectly. This is motion/color linking, not learned ReID.

These checks are conservative association heuristics, not proof of breed
correctness. Scores are not calibrated probabilities. Crops can contain
overlapping dogs, and breed detection trained on full images may behave
differently on tight crops. Tracking IDs are local video associations, **not
verified animal identities**. Color is only a lightweight appearance cue; this
tracker has no learned ReID or camera-motion correction. Crossing/overlapping
dogs can swap IDs; prolonged
occlusion or abrupt cuts can fragment tracks. Majority voting stabilizes labels
but does not prove their correctness and can propagate a wrong label throughout
a track. A breed label is never forced when all votes are unknown.

These stricter confirmation rules reduce brief/weak false positives, but can
also miss brief or low-confidence real dogs. Persistent high-scoring non-dog
objects can still confirm: temporal persistence is not proof an object is a dog.
Do not treat track count as a verified unique-animal count. A dog reappearing
after a long gap or scene cut can legitimately receive another ID. The original
user clip was deleted by the temporary-file policy, so the reported 418 IDs
cannot be reproduced on that exact video without another upload. Candidate and
discarded-candidate counts are included in per-job performance diagnostics.

Regression validation includes 4,725 synthetic frames with box jitter, weak
scores, two-second gaps and scattered one-frame noise; only one track is rendered
in that constructed case. The real five-frame GPU video check validates both
models, track consensus, MP4/audio, download and deletion. Neither proves the fix
on the user's deleted original clip. Local 300-image tracker diagnostics are at
`backend\runtime\confirmed-tracking-benchmark.json`; those are sampled images
with an assumed diagnostic FPS, not continuous-video identity ground truth.

The two-pass renderer uses a temporary job-local `track-observations.jsonl`
spool containing boxes and IDs, not frames or exported annotations. It is deleted
after processing, with the other intermediates; the API retains no tracking
history/database. A second decode adds work and is not a claimed speedup.

Tracking validation: dependency-free tests cover global matching versus exhaustive
assignment, reordered detections, a synthetic crossing, missed-frame recovery,
expired tracks, voting, unknown exclusion, tie rules, retroactive label replacement,
two-pass rendering, and spool cleanup. A live CUDA/H.264/AAC smoke check produced
one ID across five frames, all rendered with the track consensus. This does not
validate tracking accuracy on crowded real scenes. No checkpoint changes or
additional packages in the training environment are required.

Validated after this change: 15 backend tests pass, including exact clean-crop
boundaries, no breed calls for empty dog detections, original box retention,
single-label selection and unknown-breed fallback. The live CUDA smoke passed
with 5 frames, 5 dog boxes, 5 breed crops and 5 eligible breed labels; H.264/AAC,
download and deletion passed. Logs: `backend\runtime\two-stage-smoke.log`.
The restarted local API reports `dog_boxes_then_breed_crops` and `cuda:0`.
This is functional verification, not a breed accuracy evaluation.

The serving breed snapshot is `backend\assets\breed-full-best.pt`, copied and
SHA256-verified from the completed run
`models\breed_recognition\yolo12s_960_full_701515_20261007_014943\weights\best.pt`.
Run `backend\configure.py` to snapshot both models and update local settings,
preserving the existing token. Never put a token into an APK. Both checkpoints
must be available; there is no silent fallback to dog-only or CPU inference.

## Start on this PC

```powershell
powershell.exe -ExecutionPolicy Bypass -File 'C:\Work\Classes\Animal_Vocalization\backend\start-server.ps1'
```

The service binds only to `127.0.0.1:8000`. Use one process (no reload or multiple
workers), because jobs exist only in memory. Keep the PC awake and server running.

Dependencies live in the separate `backend\.venv`; the existing training
environment is not modified. The worker invokes the original environment's
Python for its already-installed Torch, Ultralytics and OpenCV. Model inference
    requires **CUDA GPU 0**, at `imgsz=1536`, with no CPU fallback. Decoding, box drawing
and H.264/AAC encoding still use CPU (limited to two threads). A SHA256-verified
copy of the completed, augmented Open Images one-class detector is at
`backend\assets\dog-openimages-aug-best.pt`. The source run is
`models\dog_recognition\yolo12s_960_openimages_aug_20261011_150702`, completed
through 100 epochs on October 11, 2026. Its best checkpoint is used, not last.pt,
the pretrained baseline, or smoke-test weights. Training checkpoints are not
modified. Running training alongside inference will share GPU memory/compute.

## Connect from your phone, away from home

Tailscale is installed on this PC. Sign in using its Windows tray application.
Install Tailscale on Android, sign in to the same private network, and enable
its VPN connection. Then run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File 'C:\Work\Classes\Animal_Vocalization\backend\connect-private.ps1'
```

This configures **private Tailscale Serve HTTPS on port 8443**, proxying localhost
8000. If prompted to enable HTTPS for your tailnet, follow the CLI's link and
rerun. It prints the exact `https://<pc>.<tailnet>.ts.net:8443` app address.
No public Funnel, router port forwarding, exit node, or LAN firewall rule is used.

In the app's PC connection panel, paste that address and the `token` value from
`C:\Work\Classes\Animal_Vocalization\backend\server.local.json`. The token is
generated randomly and never bundled into the app. Do not share that file or
commit it. Restrict tailnet access to your own devices; if inviting others, use
tailnet access controls. Both PC and phone need an internet connection, and this
PC must remain powered on. Other VPN apps can conflict with the Android VPN.

VPN login and phone installation require the owner's interaction. Remote access
is not configured merely by installing the PC client.

## Limits and temporary files

- Raw streamed upload: 250 MiB maximum; ten-minute upload timeout.
- Up to five minutes, 1–120 fps, at most 4096 pixels on either side.
- Output is aspect-ratio-preserving, even-sized, up to 1280 pixels on its long side.
- One GPU inference worker; at most two pending/completed jobs total.
- Processing timeout: 20 minutes. Start with short clips to test phone connectivity.
- Input and encoding intermediates are deleted after processing. Only the final
  MP4 remains temporarily, for 30 minutes after completion.
- Explicit deletion cancels active jobs and removes their files when safe.
- A running download holds a short file lease to avoid deletion mid-transfer.
- Shutdown clears jobs; startup clears orphan directories in the dedicated job
  root. Expiry cleanup runs every 15 seconds while the server runs. A killed/off
  PC cannot clean files until next startup. No permanent results are kept.
- No inference payloads/tokens/video content are written to HTTP access logs.

## API

All endpoints require `Authorization: Bearer <token>`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Model label, limits, configured CUDA device, image size, weights availability |
| `POST /jobs` | Raw binary video body, `application/octet-stream`; returns 202 + job |
| `GET /jobs/{id}` | Status, measured frame progress, expiry, download path |
| `GET /jobs/{id}/video` | Rendered MP4; supports range requests |
| `DELETE /jobs/{id}` | Idempotent cancellation/removal |

Status polling never invents inference completion percentages. Frame progress
comes from the container's reported frame count, capped before final encoding.
Detection count means **boxes across frames**, not distinct dogs. Breed crop
count is the number of dog regions processed, and breed-labeled count is how
many of those returned an eligible prediction. None count unique dogs.
`tracked_dog_count` counts track IDs, `breed_labeled_track_count` counts IDs with
known votes, and `consensus_labeled_count` counts rendered boxes with a winning
breed. They are not distinct-animal counts. `rendered_frames` measures second-pass
progress; `frames` retains the completed inference count. The health endpoint
reports `tracking=motion_iou_hungarian`, `breed_consensus=clip_majority_non_unknown`
and `render_passes=2`. Existing APKs ignore extra fields and remain compatible.
Confidence is a detector score, not a calibrated probability.
Video processing is intended for trusted personal uploads within your VPN, not
as a hardened public upload service. Variable-frame-rate source timing is
normalized to the decoder's reported FPS; preserve-original-timing support is a
future improvement. Audio is kept when decodable. Invalid input fails clearly.

## Validation

```powershell
Set-Location 'C:\Work\Classes\Animal_Vocalization'
& '.\backend\.venv\Scripts\python.exe' -m unittest discover -s '.\backend\tests' -v
& '.\backend\.venv\Scripts\python.exe' -m backend.tests.smoke_inference
```

The smoke check downloads a temporary official PyTorch Hub dog example and
generates a one-second dog video with audio. It verifies real GPU detections,
video encoding, audio, crop-only breed processing, retrieval and deletion.
It does not measure model accuracy.

Previous CPU-baseline validation: all eight API/lifecycle tests passed. The real smoke
check passed with five decoded frames and five dog-box detections, verified
H.264 video and AAC audio, successful retrieval, and confirmed job deletion.
The running localhost server responds to authenticated health requests and
rejects unauthenticated ones with 401. Private remote phone access is still
blocked on owner Tailscale sign-in; the installed PC client reports `NeedsLogin`.

### Validation after CUDA migration

- Backend unit/integration tests: authentication, upload limits, capacity,
  expiry, range playback, cancellation cleanup and interrupted-download leases.
- All nine backend tests pass after the migration.
- The real smoke check passed on `cuda:0` using the trained dog checkpoint:
  five decoded frames, five box detections, H.264 video with AAC audio,
  successful retrieval, and confirmed deletion. See
  `C:\Work\Classes\Animal_Vocalization\backend\runtime\gpu-smoke.log`.
- The localhost server was restarted to load the new configuration. Authenticated
  health now reports the trained model, `cuda:0`, and image size 960; unauthorized
  health requests are rejected. Health reports configuration, not continuous GPU
  readiness; the live inference smoke test exercises CUDA execution separately.
- Remote VPN/phone tests remain pending owner sign-in and device installation.

### Visual comparison with the breed detector

Both trained best checkpoints were run independently on the same 300 benchmark
frames on the RTX 4090. The gallery, rendered comparisons and reproducibility
metadata are at
`C:\Work\Classes\Animal_Vocalization\outputs\benchmarks\trained_dog_vs_breed_20261012_020825\index.html`.
The report has no ground-truth annotations and does not establish accuracy,
precision, recall, mAP or breed correctness. Its saved outputs are explicitly
requested benchmark artifacts, not retained app inference results.