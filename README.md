# Project: XXXX

## Research Question

XXXX

## Authors

- XXXX

## Status

- [ ] Data cleaning
- [ ] Summary statistics
- [ ] Main estimation
- [ ] Robustness checks
- [ ] Paper draft
- [ ] Submission

## Directory Structure

```
â”œâ”€â”€ CLAUDE.md                  # Instructions for Claude Code sessions
â”œâ”€â”€ README.md                  # This file â€” project overview
â”œâ”€â”€ Code\
â”‚   â”œâ”€â”€ _Archive\              # Superseded scripts
â”‚   â”œâ”€â”€ _Claude Logs\          # Session progress logs
â”‚   â””â”€â”€ _Claude Scripts\       # Purled .R copies of Claude's .Rmd scripts
â”‚       â””â”€â”€ _Archive\
â”œâ”€â”€ Data\
â”‚   â”œâ”€â”€ Raw\                   # Original source data (read-only)
â”‚   â””â”€â”€ Processed\             # Cleaned and constructed datasets
â”‚       â””â”€â”€ _Archive\
â”œâ”€â”€ Output\
â”‚   â”œâ”€â”€ Tables\                # LaTeX tables, CSVs
â”‚   â”‚   â””â”€â”€ _Archive\
â”‚   â””â”€â”€ Figures\               # PDFs, PNGs
â”‚       â””â”€â”€ _Archive\
â”œâ”€â”€ Literature\                # Reference papers and documentation
â”œâ”€â”€ Paper\                     # LaTeX paper and .bib files
â””â”€â”€ Admin\                     # Notes, ideas, miscellaneous
```

## Data Sources

| Dataset | Location | Description |
|---------|----------|-------------|
| XXXX    | `Data\Raw\XXXX` | XXXX |

## Pipeline

Scripts run in numbered order:

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `01_XXXX.Rmd` | `Data\Raw\XXXX` | `Data\Processed\XXXX` | XXXX |


## Paper Workflow
The active paper lives in Paper/, which is a Git clone of:

`
INSERT LINK
`
This repository is linked to Overleaf. The workflow is:

1. Edit .tex files in Paper/...
2. git add + git commit + git push to GitHub
3. Changes automatically sync to Overleaf


## Notes

XXXX
