# Git Repository Initialization Script
# For: college-football-predictor
# Owner: bharathkc05

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Git Repository Setup Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Initialize Git repository
Write-Host "[1/6] Initializing Git repository..." -ForegroundColor Yellow
git init
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to initialize Git repository" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Git repository initialized`n" -ForegroundColor Green

# Step 2: Set default branch to main
Write-Host "[2/6] Setting default branch to 'main'..." -ForegroundColor Yellow
git branch -M main
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to set branch name" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Default branch set to 'main'`n" -ForegroundColor Green

# Step 3: Add remote origin
Write-Host "[3/6] Adding remote repository..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/bharathkc05/college-football-predictor.git"
git remote add origin $remoteUrl
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Remote already exists, updating..." -ForegroundColor Yellow
    git remote set-url origin $remoteUrl
}
Write-Host "✅ Remote origin added: $remoteUrl`n" -ForegroundColor Green

# Step 4: Stage all files
Write-Host "[4/6] Staging files..." -ForegroundColor Yellow
Write-Host "   Files excluded by .gitignore:" -ForegroundColor Gray
Write-Host "   - __pycache__/" -ForegroundColor Gray
Write-Host "   - cuda_env/" -ForegroundColor Gray
Write-Host "   - *.pyc files" -ForegroundColor Gray
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to stage files" -ForegroundColor Red
    exit 1
}
Write-Host "✅ All files staged`n" -ForegroundColor Green

# Step 5: Create initial commit
Write-Host "[5/6] Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: College Football Predictor with 9 ML models

Features:
- 9 machine learning models (Score Regression + Differential Regression)
- Comprehensive preprocessing pipeline with temporal splits
- Visualization suite with 6-panel comparison plots
- Year-by-year accuracy analysis (2016-2023)
- Best model: LR_Diff_SelectKBest (66.5% accuracy)
- Dataset: NCAA FBS 2015-2023 (1,168 team-seasons, 7,679 games)"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create commit" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Initial commit created`n" -ForegroundColor Green

# Step 6: Push to GitHub
Write-Host "[6/6] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "   Target: https://github.com/bharathkc05/college-football-predictor" -ForegroundColor Gray
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to push to GitHub" -ForegroundColor Red
    Write-Host "`nPossible issues:" -ForegroundColor Yellow
    Write-Host "  1. Repository doesn't exist on GitHub yet" -ForegroundColor Yellow
    Write-Host "  2. Authentication required (use GitHub CLI or Personal Access Token)" -ForegroundColor Yellow
    Write-Host "  3. Remote repository not empty (use 'git pull origin main --allow-unrelated-histories' first)" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "   ✅ SUCCESS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nRepository successfully pushed to:" -ForegroundColor Green
Write-Host "https://github.com/bharathkc05/college-football-predictor`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Visit your repository on GitHub" -ForegroundColor White
Write-Host "  2. Add a description and topics" -ForegroundColor White
Write-Host "  3. Enable GitHub Pages (optional)" -ForegroundColor White
Write-Host "  4. Star your repository! ⭐`n" -ForegroundColor White
