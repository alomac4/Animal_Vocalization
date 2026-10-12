param(
    [string]$DatasetYaml = (Join-Path (Split-Path $PSScriptRoot -Parent) 'datasets\tsinghua_dogs\benchmark\yolo\data.yaml'),
    [string]$Name = ('yolo12s_960_existing_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
$Trainer = Join-Path $PSScriptRoot 'train_breeds.py'
$DatasetYaml = [System.IO.Path]::GetFullPath($DatasetYaml)
$Project = Join-Path $WorkspaceRoot 'weights'
$Log = Join-Path $WorkspaceRoot ('logs\' + $Name + '.terminal.log')
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing interpreter: $Python" }
if (-not (Test-Path $DatasetYaml -PathType Leaf)) { throw "Missing dataset: $DatasetYaml" }
New-Item -ItemType Directory -Path (Join-Path $WorkspaceRoot 'logs') -Force | Out-Null
Set-Location $WorkspaceRoot
Start-Transcript -Path $Log
try {
    $Host.UI.RawUI.WindowTitle = "Dog breed training - batch 4 - 960 x 960"
    Write-Host "Training using existing dataset: $DatasetYaml"
    Write-Host "Run outputs/checkpoints will be under: $Project\$Name"
    Write-Host 'Epoch summaries are printed automatically. Press Ctrl+C to interrupt.'
    & $Python -u $Trainer --data $DatasetYaml --batch 4 --project $Project --name $Name
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE. See $Log" }
} finally {
    Stop-Transcript
}