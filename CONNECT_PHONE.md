# Install BARK and connect your phone to this PC

## 1. Install the APK

The personal-test APK is:

`C:\Work\Classes\Animal_Vocalization\mobile\dist\bark-preview.apk`

Transfer it to your Android phone via USB or a private file transfer and open it.
Allow installation from that file app if Android asks. This is an ARM64 APK
(modern 64-bit Android phones), minimum Android 7, signed for private testing,
not a Play Store release. No Expo Go or Metro is needed for this installed APK.
Version 1.0.2 (version code 3) replaces the earlier prototype when signatures
match. Do not distribute this development-signed build as a production app.

## 2. Connect both devices with Tailscale

Tailscale is already installed on this PC, but requires owner sign-in. Open it
from the Windows tray and sign in. Install the official Tailscale app on Android,
sign in to the same private network, and enable the Android VPN connection.
Use only your own trusted devices in this tailnet. If sharing the network with
others, restrict access with tailnet access controls. Another active phone VPN
can prevent Tailscale from connecting.

## 3. Start the PC API

Open PowerShell and run this command. Leave its window open:

```powershell
powershell.exe -ExecutionPolicy Bypass -File 'C:\Work\Classes\Animal_Vocalization\backend\start-server.ps1'
```

If a server is already listening on port 8000, use that running instance rather
than starting a second one. The current code requires both model checkpoints
and a working CUDA environment; model inference uses the RTX 4090. Keep this PC
awake and connected to the internet. Frame decoding/video encoding still use CPU.

## 4. Enable the private HTTPS address

In another PowerShell window run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File 'C:\Work\Classes\Animal_Vocalization\backend\connect-private.ps1'
```

If Tailscale asks you to enable HTTPS, follow its displayed admin URL and rerun.
The script prints the exact phone-app server URL, for example:

```text
https://YOUR-PC.YOUR-TAILNET.ts.net:8443
```

Use the address actually printed by the script, not this example and not
`localhost`, `127.0.0.1`, your Wi-Fi address, or a raw Tailscale IP. This setup
uses **Tailscale Serve**, which is private to your tailnet. Do not enable Funnel
or forward router ports. Both devices must be connected to Tailscale for remote
access. The PC API itself listens only on localhost.

## 5. Configure BARK on the phone

1. Open BARK and expand **PC connection** (the control below the centered logo).
2. Paste the private HTTPS URL printed above, including port `8443`.
3. On the PC open
   `C:\Work\Classes\Animal_Vocalization\backend\server.local.json`.
4. Copy only its `token` value into the app's API-token field. Do not share the
   config file, paste it into chat, or put the token in a public link.
5. Tap **Test & save connection**. The app saves the connection in SecureStore.
6. Choose a short video and tap **Analyze video**. Keep the app open.
7. When ready, load the preview and tap **Download video** to choose a writable
   folder, or use **Share video**. Saving the file gives you the rendered MP4,
   not annotation data. The server deletes its temporary result after 30 minutes.

Pipeline: dog detection first; breed detector only on each detected dog's crop.
Only dog IDs (such as `Dog #1`) are rendered on the video. The **Dog statistics**
section under the preview lists each matching ID and its final non-unknown voted
breed, or `Breed unknown`. Confirmed dog boxes stay even when breed is unknown.
Possible breeds and tracking identities can be wrong, including on mixed breeds.
Statistics are temporary app/API metadata, not embedded in the downloaded MP4
or saved as an annotation file. Save any information you need before starting over.
Previously rendered videos cannot have their old burned-in breed text removed;
analyze them again after updating the server and APK.

## Troubleshooting

- **Incorrect API token:** recopy the current local config token, without quotes.
- **Connection timeout:** confirm PC awake, server running, Tailscale connected
  on both devices, and the exact HTTPS name/port is entered. Confirm Serve with:
  `& 'C:\Program Files\Tailscale\tailscale.exe' serve status`.
- **Breed weights missing / old pipeline:** run
  `& 'C:\Work\Classes\Animal_Vocalization\backend\.venv\Scripts\python.exe' 'C:\Work\Classes\Animal_Vocalization\backend\configure.py'`
  then restart the API to load the updated settings. Stop only the API, not training.
- **CUDA unavailable/out of memory:** verify NVIDIA drivers and available GPU
  memory. There is no CPU fallback. Training/inference compete for GPU resources.
- **PC busy:** finish/remove the previous jobs; capacity is intentionally bounded.
- **Download folder denied:** choose a writable subfolder instead of the root
  Downloads directory, or use Share. No broad storage permissions are required.
- **Result expired:** upload again. There is no persistent inference history.

VPN owner sign-in, installing the APK, and native phone tests require your
interaction. PC-side tests alone cannot verify your phone's networking or folder
permissions. Account login has not been performed automatically.

Official reference: Tailscale documentation, “Tailscale Serve” and
“tailscale serve command”; Serve publishes HTTPS inside a tailnet, unlike Funnel.