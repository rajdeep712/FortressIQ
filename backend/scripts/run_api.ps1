$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8000