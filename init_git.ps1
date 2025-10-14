# Git Repository Setup - Manual Commands
# For: college-football-predictor
# Owner: bharathkc05

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Git Repository Setup Guide" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Please run the following commands manually:`n" -ForegroundColor Yellow

Write-Host "Step 1: Initialize Git repository" -ForegroundColor Green
Write-Host "git init`n" -ForegroundColor White

Write-Host "Step 2: Set default branch to main" -ForegroundColor Green
Write-Host "git branch -M main`n" -ForegroundColor White

Write-Host "Step 3: Add remote repository" -ForegroundColor Green
Write-Host "git remote add origin https://github.com/bharathkc05/college-football-predictor.git`n" -ForegroundColor White

Write-Host "Step 4: Stage all files" -ForegroundColor Green
Write-Host "git add .`n" -ForegroundColor White

Write-Host "Step 5: Create initial commit" -ForegroundColor Green
Write-Host 'git commit -m "Initial commit: College Football Predictor"' -ForegroundColor White
Write-Host ""

Write-Host "Step 6: Push to GitHub" -ForegroundColor Green
Write-Host "git push -u origin main`n" -ForegroundColor White

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Note: .gitignore will automatically exclude:" -ForegroundColor Yellow
Write-Host "  - __pycache__/" -ForegroundColor Gray
Write-Host "  - cuda_env/" -ForegroundColor Gray
Write-Host "  - *.pyc files" -ForegroundColor Gray
Write-Host "========================================`n" -ForegroundColor Cyan
