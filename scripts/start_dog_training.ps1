param(
    [string]$DataRoot = 'E:\Datasets\OpenImagesDogs\yolo_70_15_15_seed42',
    [string]$Name = ('yolo12s_960_openimages_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
Set-Location -LiteralPath $Root
& $Python -u (Join-Path $PSScriptRoot 'dog_training_pipeline.py') --data-root $DataRoot --name $Name --single-dog-augmentation
if ($LASTEXITCODE -ne 0) { throw "Dog training pipeline failed with exit code $LASTEXITCODE" }