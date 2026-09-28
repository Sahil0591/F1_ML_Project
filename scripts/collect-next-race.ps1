param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pythonPath = Join-Path $resolvedRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Install the project virtual environment before starting collection.'
}
$logDirectory = Join-Path $resolvedRoot 'data\raw\prospective_scheduler\logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory ((Get-Date -Format 'yyyy-MM-dd') + '.log')
Push-Location -LiteralPath $resolvedRoot
try {
    & $pythonPath -m f1_ml_predictor collect-next-race 2>&1 | Out-File -LiteralPath $logPath -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Collection tick failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
