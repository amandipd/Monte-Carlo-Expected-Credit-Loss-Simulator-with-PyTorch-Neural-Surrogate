#Requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Running unit tests..."
poetry run python -m pytest -q -m "not integration"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Running end-to-end pipeline validation..."
poetry run python scripts/validate_pipeline.py
exit $LASTEXITCODE
