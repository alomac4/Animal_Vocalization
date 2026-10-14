param([int]$ExpectedProcessId)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$jobs = Join-Path $PSScriptRoot 'runtime\jobs'
$deadline = (Get-Date).AddHours(1)
Write-Output 'Waiting for all temporary results/jobs to be released. Existing results will not be deleted early.'
while ((Get-Date) -lt $deadline) {
    $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if (-not $listener -or $listener.OwningProcess -ne $ExpectedProcessId) {
        Write-Output 'Server ownership changed; no process was stopped.'
        exit 0
    }
    $worker = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -like "*${root}\backend\worker.py*"
    }
    $folders = @(Get-ChildItem -LiteralPath $jobs -Directory -ErrorAction SilentlyContinue)
    if ($folders.Count -eq 0 -and -not $worker) {
        # Recheck before stopping the one server process, never a training process.
        Start-Sleep -Seconds 2
        if (@(Get-ChildItem -LiteralPath $jobs -Directory -ErrorAction SilentlyContinue).Count -ne 0) { continue }
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ExpectedProcessId"
        if (-not $process -or $process.CommandLine -notlike '*uvicorn backend.app:create_app*') {
            Write-Output 'Expected API process not found; restart cancelled.'
            exit 0
        }
        Stop-Process -Id $ExpectedProcessId
        Start-Sleep -Seconds 2
        $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        $newProcess = Start-Process -FilePath $python -ArgumentList '-m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log' -WorkingDirectory $root -RedirectStandardOutput (Join-Path $PSScriptRoot 'runtime\server.log') -RedirectStandardError (Join-Path $PSScriptRoot 'runtime\server.err.log') -PassThru
        Write-Output "Restarted idle API. Launcher PID: $($newProcess.Id)"
        $config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server.local.json') -Raw | ConvertFrom-Json
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Headers @{Authorization="Bearer $($config.token)"} -TimeoutSec 3
                $health | Select-Object device,imgsz,tracking,track_buffer_seconds,track_new_conf,track_min_hits | ConvertTo-Json
                if ($health.tracking -notin @('confirmed_motion_color_hungarian','confirmed_ocsort_sequential','confirmed_botsort_sequential','confirmed_fasttrack_sequential')) { throw 'Unexpected tracking version' }
                exit 0
            } catch { Start-Sleep -Seconds 1 }
        }
        throw 'Restarted API did not pass health verification'
    }
    Start-Sleep -Seconds 15
}
Write-Output 'No idle window within one hour; existing server left running.'