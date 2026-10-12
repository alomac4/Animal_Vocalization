$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path (Join-Path $PSScriptRoot 'server.local.json'))) {
    & $python (Join-Path $PSScriptRoot 'configure.py')
    if ($LASTEXITCODE -ne 0) { throw 'Server configuration failed' }
}
Set-Location -LiteralPath $root
# One process owns the in-memory registry. Never use --reload or multiple workers.
& $python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log