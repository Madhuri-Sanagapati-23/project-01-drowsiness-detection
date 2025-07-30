Write-Host "Activating Virtual Environment..." -ForegroundColor Green
& ".venv\Scripts\Activate.ps1"
Write-Host "Virtual Environment Activated!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the drowsiness detection system:" -ForegroundColor Yellow
Write-Host "  python main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor Cyan
Write-Host "" 