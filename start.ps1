<#
.SYNOPSIS
    Starts both the Tasbir FastAPI backend (port 8000) and Next.js frontend (port 3000).
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Starting Tasbir Backend (http://localhost:8000)..." -ForegroundColor Cyan
$backend = Start-Process -FilePath "$Root\backend\.venv\Scripts\uvicorn.exe" `
    -ArgumentList "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory "$Root\backend" `
    -PassThru

Write-Host "Starting Tasbir Frontend (http://localhost:3000)..." -ForegroundColor Green
$frontend = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory "$Root\frontend" `
    -PassThru

Write-Host ""
Write-Host "Tasbir is running:" -ForegroundColor Yellow
Write-Host "  Web App : http://localhost:3000"
Write-Host "  API Docs: http://localhost:8000/docs"
Write-Host "Press Ctrl+C to stop both servers." -ForegroundColor DarkGray

try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    if (!$backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if (!$frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
}
