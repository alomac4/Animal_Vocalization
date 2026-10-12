param(
    [string]$DataRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'datasets\tsinghua_dogs'),
    [double]$Batch = 4,
    [switch]$SkipInvalidAnnotations
)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw 'Python environment missing. Follow the environment setup in README.md first.'
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$Metadata = Join-Path $DataRoot 'metadata'
$Archives = Join-Path $DataRoot 'original'
$Images = Join-Path $Archives 'high-resolution'
$Converted = Join-Path $DataRoot 'yolo_breeds'

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Python -Arguments @(
    (Join-Path $PSScriptRoot 'download_dataset.py'),
    '--metadata', $Metadata, '--images', $Archives, '--extract'
)
$DatasetYaml = Join-Path $Converted 'data.yaml'
if (-not (Test-Path $DatasetYaml)) {
    $PrepareArgs = @(
        (Join-Path $PSScriptRoot 'prepare_dataset.py'),
        '--annotations', (Join-Path $Metadata 'High-Annotations'),
        '--splits', (Join-Path $Metadata 'TrainAndValList'),
        '--images', $Images, '--output', $Converted
    )
    if ($SkipInvalidAnnotations) {
        $PrepareArgs += '--skip-invalid'
    }
    Invoke-Python -Arguments $PrepareArgs
}
Invoke-Python -Arguments @(
    (Join-Path $PSScriptRoot 'train_breeds.py'), '--data', $DatasetYaml,
    '--batch', $Batch.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--check-only'
)
Invoke-Python -Arguments @(
    (Join-Path $PSScriptRoot 'train_breeds.py'), '--data', $DatasetYaml,
    '--batch', $Batch.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--smoke-test'
)
Invoke-Python -Arguments @(
    (Join-Path $PSScriptRoot 'train_breeds.py'), '--data', $DatasetYaml,
    '--batch', $Batch.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)