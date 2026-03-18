# NeuSight Reproduction Baseline Log

## Worktree
- Path: `/work/NeuSight/.worktrees/paper-repro-a100`
- Branch: `paper-repro-a100`

## Git
- Base branch at creation: `main`
- Worktree branch HEAD before this log commit: `ab91745`

## Repository Baseline Checks
- `scripts/asplos/data/predictor/MLP_WAVE`: present
- `scripts/asplos/label`: present
- `scripts/asplos/results`: present
- `scripts/asplos/summary`: present
- `scripts/asplos/run_pred_nvidia_neusight.py`: present
- Baseline command result: `baseline-ok`

## Environment Baseline
System `python3` currently lacks the runtime dependencies needed for script execution:
- `pandas`: missing
- `torch`: missing
- `transformers`: missing

A previous attempt to create a temporary venv and run `pip install -e .` started downloading large CUDA-enabled dependencies. That path was stopped because it would install a fresh heavyweight stack before verifying whether a lighter reproduction route was possible.

## Verification Strategy in This Repository
Because there is no discovered automated test suite, verification will use command-level checks:
1. Baseline path existence checks.
2. `summarize.py` / `table.py` regeneration checks once dependencies are available.
3. File existence checks for regenerated `summary/*.csv` outputs.
4. Explicit extraction of A100 rows from regenerated summaries.
5. Final report comparing regenerated values against repository reference outputs.
