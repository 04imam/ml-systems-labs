$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m pip install -r requirements.txt
python .\src\run_lab7.py
if ($LASTEXITCODE -ne 0) { throw 'Lab 7 run failed' }
Write-Host 'Lab 7 complete. Inspect report.md and results folder.' -ForegroundColor Green
