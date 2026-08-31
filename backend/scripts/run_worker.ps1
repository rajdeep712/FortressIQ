$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")

& ".\.venv\Scripts\python.exe" -m arq app.ingestion.worker.WorkerSettings