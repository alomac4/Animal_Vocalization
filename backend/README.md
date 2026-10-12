# BARK PC inference server

Authenticated FastAPI service. No database, accounts, result history, JSON
annotation download, or MCP server. It burns dog boxes into a playable H.264 MP4
and preserves the first input audio stream as AAC when available.

## Two-stage inference

Phone install/connect instructions are also in
`C:\Work\Classes\Animal_Vocalization\CONNECT_PHONE.md`.

1. The trained dog-only detector finds dogs in each frame on `cuda:0`.
2. Each valid dog box is clipped to the frame and copied as an unannotated crop.
3. The trained 130-class breed detector runs only on that crop, on the same GPU
   at image size 960. It never searches the full frame for additional boxes.
4. The final video uses the **original dog detector box**. A possible breed label
   is added if the highest-scoring eligible crop detection scores at least 0.25,
   includes the crop center, and covers at least 20% of the crop area. Otherwise
   the dog box remains, labeled `Dog <score> | breed unknown`.

These checks are conservative association heuristics, not proof of breed
correctness. Scores are not calibrated probabilities. Crops can contain
overlapping dogs, and breed detection trained on full images may behave
differently on tight crops. There is no tracking or identity assignment; each
frame is inferred independently. Breed labels may vary between frames. No label
is forced and no dog detection is removed when the breed model returns nothing.

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
requires **CUDA GPU 0**, at `imgsz=960`, with no CPU fallback. Decoding, box drawing
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
Confidence is a detector score, not a calibrated probability. No identity inference.
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