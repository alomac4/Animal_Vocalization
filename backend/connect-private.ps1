$ErrorActionPreference = 'Stop'
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
if (-not (Test-Path $tailscale)) { throw 'Install Tailscale on this PC first.' }
$raw = & $tailscale status --json
if ($LASTEXITCODE -ne 0) { throw 'Open Tailscale and sign in before running this script.' }
$status = $raw | ConvertFrom-Json
if ($status.BackendState -ne 'Running') { throw 'Open Tailscale and sign in to the same account as your phone.' }
# Private tailnet HTTPS only. Do not use Funnel or router port forwarding.
& $tailscale serve --bg --yes --https=8443 http://127.0.0.1:8000
if ($LASTEXITCODE -ne 0) { throw 'Tailscale Serve needs HTTPS enabled. Follow the URL printed above, then rerun.' }
$name = $status.Self.DNSName.TrimEnd('.')
Write-Output "Phone app server URL: https://${name}:8443"
Write-Output 'Use the API token from backend\server.local.json in the app connection panel.'