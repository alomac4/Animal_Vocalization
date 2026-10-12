# BARK PC inference server

Authenticated FastAPI service. No database, accounts, result history, JSON
annotation download, or MCP server. It burns dog boxes into a playable H.264 MP4
and preserves the first input audio stream as AAC when available.

## Start on this PC

```powershell
powershell.exe -ExecutionPolicy Bypass -File 'C:\Work\Classes\Animal_Vocalization\backend\start-server.ps1'
```

The service binds only to `127.0.0.1:8000`. Use one process (no reload or multiple
workers), because jobs exist only in memory. Keep the PC awake and server running.

Dependencies live in the separate `backend\.venv`; the existing training
environment is not modified. The CPU worker invokes the original environment's
Python for its already-installed Torch, Ultralytics and OpenCV, with CUDA hidden
and two CPU threads. A stable copy of the pretrained 80-class YOLO12s checkpoint
is in `backend\assets\baseline-yolo12s.pt`; only its `dog` class is used. This is
**not the model currently training**, and smoke-test weights are never selected.
Training checkpoints/configuration are not touched.

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
- One CPU inference worker; at most two pending/completed jobs total.
- Processing timeout: 20 minutes. Long videos may time out on CPU; start short.
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
| `GET /health` | Model label, limits, CPU device, weights availability |
| `POST /jobs` | Raw binary video body, `application/octet-stream`; returns 202 + job |
| `GET /jobs/{id}` | Status, measured frame progress, expiry, download path |
| `GET /jobs/{id}/video` | Rendered MP4; supports range requests |
| `DELETE /jobs/{id}` | Idempotent cancellation/removal |

Status polling never invents inference completion percentages. Frame progress
comes from the container's reported frame count, capped before final encoding.
Detection count means **boxes across frames**, not distinct dogs. Confidence is
a detector score, not a calibrated probability. No breed or identity inference.
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
generates a one-second dog video with audio. It verifies real CPU detections,
video encoding, audio, retrieval and deletion. It does not measure model accuracy.

Current local validation: all eight API/lifecycle tests pass. The real smoke
check passed with five decoded frames and five dog-box detections, verified
H.264 video and AAC audio, successful retrieval, and confirmed job deletion.
The running localhost server responds to authenticated health requests and
rejects unauthenticated ones with 401. Private remote phone access is still
blocked on owner Tailscale sign-in; the installed PC client reports `NeedsLogin`.

### Latest verified checks

- Eight backend unit/integration tests: authentication, upload limits, capacity,
  expiry, range playback, cancellation cleanup and interrupted-download leases.
- Real CPU smoke run: five frames, five dog box detections, H.264/AAC MP4,
  result download and deletion.
- Server health tested at localhost 8000 with an authenticated request.
- Remote VPN/phone tests remain pending owner sign-in and device installation.