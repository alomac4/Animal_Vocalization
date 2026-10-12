param(
    [string]$DataRoot = 'E:\Datasets\TsinghuaDogs',
    [string]$Name = ('yolo12s_960_full_701515_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
$Pipeline = Join-Path $PSScriptRoot 'full_training_pipeline.py'
$Host.UI.RawUI.WindowTitle = 'Full dog dataset - 70/15/15 - batch 4 - 960 x 960'
Set-Location $WorkspaceRoot
Write-Host "Dataset: $DataRoot"
Write-Host "Checkpoints: $WorkspaceRoot\weights\$Name\weights"
Write-Host "Live pipeline log: $WorkspaceRoot\logs\$Name.pipeline.log"
Write-Host 'Keep this terminal open. Progress and epoch summaries appear automatically.'
& $Python -u $Pipeline --data-root $DataRoot --name $Name
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }