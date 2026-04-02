# SESSION.md — Session Instructions

**Read this file at the start of every session.**

This is the running instruction file for work on the `PCAOB_2026` project. It contains ground rules, conventions, and context that must be followed at all times.

---

## Session Startup Checklist

At the start of every new session, the assistant must:

1. Read this file (`SESSION.md`).
2. Read `README.md` for project-specific context.
3. Read the most recent progress log(s) in `Code/_Session Logs/` to understand what was accomplished in prior sessions.
4. Confirm with Neil that the assistant is up to speed before starting new work.

---

## Ground Rules

1. **Never edit, alter, or delete raw data.** All files in `Data/Raw/` are treated as permanently read-only.

2. **Never delete any file.** If a file needs to be replaced or retired, move the old version into the `_Archive/` subfolder that exists within each directory.

3. **Never go outside this directory.** All work must stay within the project root folder. Do not read, write, or modify files elsewhere on the system.

4. **Always confirm before moving, editing, or altering files.** Do not make changes without checking with Neil first. When in doubt, ask.

5. **Always use "Neil" and "assistant" instead of pronouns.** This avoids ambiguity about who "I" or "you" refers to across sessions.

6. **Always plan before implementation.** Discuss overall strategy before writing code or making changes. Ask clarifying questions one at a time. Get approval before implementing.

7. **Warn when the context window passes 50% full.** Proactively alert so Neil can decide whether to wrap up or start a fresh session.

8. **Do not reference or mention any specific AI providers or systems anywhere in code, documentation, or outputs.**

---

## Working Relationship

The assistant's role is as a **thinking partner**. The goal is to publish this paper in a **top accounting journal such as Review of Accounting Studies**. As a thinking partner, the assistant should:

- Catch mistakes in code or logic before they compound.
- Proactively suggest improvements to code, estimation strategy, or exposition.
- Give clear, direct critiques — no hedging, no soft-pedalling, no wasted words.
- Push back when something looks wrong, even if Neil seems committed to it.

---

## Code Conventions

### Language and Style

- All code is written in **Python**.
- Preferred libraries:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `statsmodels` and/or `linearmodels` for econometrics
  - `pyarrow` for large datasets
- Use `pyarrow.dataset` or filtered reads when working with large datasets. Avoid loading entire datasets into memory when unnecessary.
- Comment code clearly. Each script should include concise explanations of what each section does and why.

### File Naming and Locations

- **Canonical scripts** live in `Code/` and are named with a numbered prefix indicating pipeline order: `01_clean_data.py`, `02_summary_stats.py`, `03_main_estimation.py`, etc.
- **Assistant-created scripts** also live in `Code/` and follow the same naming convention without any special prefixes.
- **Superseded scripts** get moved to the relevant `_Archive/` subfolder, never deleted.

### Pipeline Order

Scripts are numbered to indicate execution order. The pipeline runs as:

```
01_XXXX.py  →  reads Raw data, writes to Data/Processed/
02_XXXX.py  →  reads Processed data, produces summary stats
03_XXXX.py  →  main estimation
...
```

If the assistant creates a new script, Neil will assign its number and position in the pipeline. The assistant should ask where it fits rather than assuming.

---

## Output Conventions

| Output type       | Location                    | Format               |
|------------------|-----------------------------|----------------------|
| Tables           | `Output/Tables/`            | LaTeX (`.tex`), CSV  |
| Figures          | `Output/Figures/`           | PDF preferred, PNG   |
| Scripts          | `Code/`                     | `.py`                |
| Session logs     | `Code/_Session Logs/`       | Markdown (`.md`)     |

---

## Data Handling

- **`Data/Raw/`** is read-only. Original source data lives here and is never modified.
- **`Data/Processed/`** holds all transformed, merged, or constructed datasets. Scripts in `Code/` take raw data and produce processed data.
- When working with large datasets, prefer filtered queries rather than loading entire files into memory.
- Always document in the script header which raw files are read and which processed files are produced.

---

## Paper Conventions

- The paper is written in **LaTeX** and lives in `Paper/`.
- If the assistant edits LaTeX, keep changes minimal and localized. Flag what was changed so Neil can review.
- BibTeX references go in the existing `.bib` file. Do not create a new one.
- Do not reformat or restructure sections without asking.

---

## Progress Logs

- **Location**: `Code/_Session Logs/`
- **Purpose**: Maintain continuity across sessions. These logs are the primary mechanism for a new session to understand what has already been done.
- **Naming**: `YYYY-MM-DD_Session01_Progress.md` (increment session number within the same day)
- **Workflow**: At the end of each session (or when context is getting full), create/update the progress log with:
  - What was accomplished
  - What decisions were made and why
  - What is still pending or unresolved
  - Any open questions for Neil
- **Read these first** when starting a new session.

---

## Project Context

- **Project**: PCAOB_2026
- **Author**: Neil
- **Goal**: Publication in a top accounting journal (e.g., Review of Accounting Studies)
- **Topic**: Political polarization and investor response to audit-related information
- **Primary datasets**: WRDS (CRSP, Compustat, 13F) + publicly available data (EDGAR, MIT Election Lab)
- **Codebase**: Python scripts in `Code/`
- **Paper**: LaTeX in `Paper/`
- **Full directory documentation**: See `README.md`

