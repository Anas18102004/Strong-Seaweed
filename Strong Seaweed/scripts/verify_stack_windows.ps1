param()

$ErrorActionPreference = "Continue"

Write-Host "== BlueWeave stack verification ==" -ForegroundColor Cyan

function Test-Endpoint {
  param(
    [string]$Name,
    [string]$Url
  )
  try {
    $res = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 8
    Write-Host ("[OK]   {0} -> {1}" -f $Name, $res.StatusCode) -ForegroundColor Green
  } catch {
    Write-Host ("[FAIL] {0} -> {1}" -f $Name, $_.Exception.Message) -ForegroundColor Yellow
  }
}

Test-Endpoint -Name "Frontend" -Url "http://127.0.0.1:8080"
Test-Endpoint -Name "Backend health" -Url "http://127.0.0.1:4000/health"
Test-Endpoint -Name "Agent gateway health" -Url "http://127.0.0.1:8101/health"
Test-Endpoint -Name "Model API health" -Url "http://127.0.0.1:8000/health"

Write-Host "`nIf some checks failed, start missing services using README instructions." -ForegroundColor DarkCyan
