param()

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

Write-Host "== BlueWeave + Strong Seaweed bootstrap (Windows) ==" -ForegroundColor Cyan

function Ensure-FileFromExample {
  param(
    [string]$Path,
    [string]$ExamplePath
  )
  if (!(Test-Path $Path) -and (Test-Path $ExamplePath)) {
    Copy-Item $ExamplePath $Path -Force
    Write-Host "Created $Path from example." -ForegroundColor Green
  }
}

# Frontend install
Push-Location (Join-Path $RootDir "blue-weave-aqua-main")
npm install
Pop-Location

# Backend install + env
Push-Location (Join-Path $RootDir "blue-weave-aqua-main\server")
Ensure-FileFromExample -Path ".env" -ExamplePath ".env.example"
npm install
Pop-Location

# Agent gateway venv + deps
Push-Location (Join-Path $RootDir "blue-weave-aqua-main\server\agents_python")
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
.\.venv\Scripts\python -m pip install -r requirements.txt
Pop-Location

# Model API venv + deps
Push-Location (Join-Path $RootDir "Strong-Seaweed-main")
Ensure-FileFromExample -Path ".env" -ExamplePath ".env.example"
if (!(Test-Path ".venv_model_api")) {
  python -m venv .venv_model_api
}
.\.venv_model_api\Scripts\python -m pip install -r requirements-model-api.txt
Pop-Location

Write-Host "`nBootstrap complete." -ForegroundColor Green
Write-Host "Start services in 4 terminals (model API, agent gateway, backend, frontend) as documented in README.md"
