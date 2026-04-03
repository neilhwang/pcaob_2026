# sync_overleaf.ps1
# =================
# Syncs Paper/ and Output/ from the main project to the pcaob_paper
# Overleaf-linked repo, then commits and pushes to GitHub.
#
# Run this from the project root after any session that changes LaTeX
# files or regenerates tables/figures:
#
#   .\sync_overleaf.ps1
#
# Overleaf will then pick up the changes via Menu -> GitHub -> Pull.

$PROJECT = "C:\Users\neilh\documents\pcaob_2026"
$PAPER   = "C:\Users\neilh\documents\pcaob_paper"

Write-Host "Syncing paper files to Overleaf repo..." -ForegroundColor Cyan

# LaTeX source
Copy-Item "$PROJECT\Paper\draft.tex"      "$PAPER\Paper\draft.tex"      -Force
Copy-Item "$PROJECT\Paper\references.bib" "$PAPER\Paper\references.bib" -Force

# Tables
Copy-Item "$PROJECT\Output\Tables\*.tex"  "$PAPER\Output\Tables\" -Force

# Figures (if any exist)
$figSrc = "$PROJECT\Output\Figures"
$figDst = "$PAPER\Output\Figures"
if (Test-Path $figSrc) {
    if (-not (Test-Path $figDst)) { New-Item -ItemType Directory -Path $figDst | Out-Null }
    Copy-Item "$figSrc\*" $figDst -Recurse -Force
}

# Commit and push
Set-Location $PAPER
git add .

# Only commit if there are actual changes
$status = git status --porcelain
if ($status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "Sync $timestamp"
    git push
    Write-Host "Done. Changes pushed to GitHub." -ForegroundColor Green
    Write-Host "In Overleaf: Menu -> GitHub -> Pull to see updates." -ForegroundColor Yellow
} else {
    Write-Host "Nothing to sync - Overleaf repo already up to date." -ForegroundColor Green
}

# Return to project root
Set-Location $PROJECT
