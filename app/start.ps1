# Pocket Me — start the AI Brain backend
#   Run from anywhere:   .\start.ps1
#
# Uses "python -m uvicorn" on purpose: pip installs uvicorn.exe into a Scripts
# folder that is usually not on PATH, so a bare "uvicorn" fails with
# "'uvicorn' is not recognized".

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:GEMINI_API_KEY) {
    Write-Host ""
    Write-Host "GEMINI_API_KEY is not set in this terminal." -ForegroundColor Yellow
    Write-Host "The server will start, but every AI request will fail until you set it."
    Write-Host ""
    Write-Host '  $env:GEMINI_API_KEY = "your-key"      # this terminal only'
    Write-Host '  setx GEMINI_API_KEY "your-key"        # permanent, then open a NEW terminal'
    Write-Host ""
    Write-Host "Get a key at https://aistudio.google.com/apikey"
    Write-Host ""
    $answer = Read-Host "Start anyway? (y/N)"
    if ($answer -ne "y") { Write-Host "Cancelled."; exit 1 }
} else {
    Write-Host "GEMINI_API_KEY: set (length $($env:GEMINI_API_KEY.Length))" -ForegroundColor Green
}

# Fail early with a clear message if the port is already taken.
$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $pid8000 = $busy[0].OwningProcess
    $proc = Get-Process -Id $pid8000 -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "Port 8000 is already in use by PID $pid8000 ($($proc.ProcessName))." -ForegroundColor Red
    Write-Host "Stop it first (Ctrl+C in its window), or run:  Stop-Process -Id $pid8000"
    exit 1
}

Write-Host "Starting Pocket Me brain on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Health check: http://127.0.0.1:8000/"
Write-Host "Models your key can call: http://127.0.0.1:8000/api/v1/brain/models"
Write-Host ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
