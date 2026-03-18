# NeuSight Paper Reproduction and A100 GPT-3 Prefill/Decode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce the paper-supported NeuSight ASPLOS workflow from the local repository first, then make the smallest viable code changes needed to obtain A100 GPT-3 prefill and decode results for the user’s requested parameters.

**Architecture:** First treat the existing ASPLOS scripts, labels, predictors, results, and summaries as the ground-truth reproduction path and verify that the repository can regenerate the expected NVIDIA outputs. Then extend the current `inf/train` tracing and prediction pipeline with a minimal `prefill/decode` path, reusing the existing predictor, parser, aggregator, and A100 device configuration wherever possible.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers, pandas, existing NeuSight scripts under `scripts/asplos`, Git worktrees.

---

### Task 1: Capture the reproducible local baseline

**Files:**
- Create: `docs/reports/2026-03-18-neusight-reproduction-log.md`
- Modify: `README.md` only if an actual environment/documentation gap is discovered
- Test: no repo test suite exists; use command-level smoke verification documented in the report

**Step 1: Write the failing test**

Record the baseline checks we expect to succeed once the environment is usable:

```text
- repository worktree exists and is isolated
- ASPLOS data/predictor/label directories are present
- NVIDIA reproduction scripts are discoverable
- summarize.py can regenerate summary.csv
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
python3 - <<'PY'
import os
required = [
    'scripts/asplos/data/predictor/MLP_WAVE',
    'scripts/asplos/label',
    'scripts/asplos/results',
    'scripts/asplos/summary',
    'scripts/asplos/run_pred_nvidia_neusight.py',
]
missing = [p for p in required if not os.path.exists(p)]
assert not missing, missing
print('baseline-ok')
PY
```

Expected: if anything is missing, fail with the missing path list.

**Step 3: Write minimal implementation**

Create `docs/reports/2026-03-18-neusight-reproduction-log.md` and record:
- worktree path
- current branch and commit
- discovered environment blockers such as missing `torch`
- which commands will be treated as verification in place of a formal test suite

**Step 4: Run test to verify it passes**

Run the same baseline command again and append the result to the report.

Expected: `baseline-ok`

**Step 5: Commit**

```bash
git add docs/reports/2026-03-18-neusight-reproduction-log.md
git commit -m "docs: add reproduction baseline log"
```

### Task 2: Re-run the paper-supported NVIDIA NeuSight prediction workflow

**Files:**
- Modify: `scripts/asplos/results/**` only through script output regeneration
- Modify: `scripts/asplos/summary/summary.csv`
- Modify: `scripts/asplos/summary/nvidia_inf.csv`
- Modify: `scripts/asplos/summary/nvidia_train.csv`
- Modify: `docs/reports/2026-03-18-neusight-reproduction-log.md`
- Test: regenerated `scripts/asplos/summary/*.csv`

**Step 1: Write the failing test**

Define success as being able to regenerate summary outputs from local scripts:

```text
- summarize.py finishes successfully
- summary.csv is regenerated
- A100 rows are present in summary.csv or derived table CSVs
```

**Step 2: Run test to verify it fails**

Force a fresh regeneration in a temporary copy of the summary outputs:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100/scripts/asplos
rm -f summary/summary.csv summary/nvidia_inf.csv summary/nvidia_train.csv
python3 summarize.py
python3 table.py
test -f summary/summary.csv
```

Expected: fail if summarize/table cannot run in the current environment.

**Step 3: Write minimal implementation**

If the commands fail, make the smallest necessary compatibility fix in the exact failing file(s), for example:
- import/path fixes in `scripts/asplos/summarize.py`
- import/path fixes in `scripts/asplos/table.py`
- environment-agnostic path handling in `scripts/asplos/run_pred_nvidia_neusight.py`

Do not rewrite the workflow.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100/scripts/asplos
python3 summarize.py
python3 table.py
python3 - <<'PY'
import pandas as pd
s = pd.read_csv('summary/summary.csv')
print('rows', len(s))
print('has_a100', ((s['device'] == 'NVIDIA_A100-PCIE-40GB') | (s['device'] == 'NVIDIA_A100_80GB_PCIe')).any())
PY
```

Expected: non-zero rows and `has_a100 True`.

**Step 5: Commit**

```bash
git add scripts/asplos docs/reports/2026-03-18-neusight-reproduction-log.md
git commit -m "fix: restore paper summary reproduction workflow"
```

### Task 3: Compare regenerated paper outputs against the repository’s reference data

**Files:**
- Create: `docs/reports/2026-03-18-neusight-paper-reproduction.md`
- Modify: `docs/reports/2026-03-18-neusight-reproduction-log.md`
- Test: `docs/reports/2026-03-18-neusight-paper-reproduction.md`

**Step 1: Write the failing test**

Define the report sections that must exist:

```markdown
## Inputs
## Commands Run
## Regenerated Files
## A100 Results
## Delta vs Reference
## Reproduction Verdict
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
python3 - <<'PY'
from pathlib import Path
p = Path('docs/reports/2026-03-18-neusight-paper-reproduction.md')
assert p.exists(), 'missing report'
text = p.read_text()
for heading in ['## Inputs','## Commands Run','## Regenerated Files','## A100 Results','## Delta vs Reference','## Reproduction Verdict']:
    assert heading in text, heading
print('report-ok')
PY
```

Expected: fail because the report does not exist yet.

**Step 3: Write minimal implementation**

Create the report and include:
- which scripts were run
- which outputs were regenerated
- extracted A100 rows from `summary/summary.csv`, `summary/nvidia_inf.csv`, and `summary/nvidia_train.csv`
- a comparison against the repository reference summary values
- a concise verdict: reproduced / approximately reproduced / blocked

**Step 4: Run test to verify it passes**

Re-run the report validation command above.

Expected: `report-ok`

**Step 5: Commit**

```bash
git add docs/reports/2026-03-18-neusight-paper-reproduction.md docs/reports/2026-03-18-neusight-reproduction-log.md
git commit -m "docs: add paper reproduction report"
```

### Task 4: Add a GPT-3 175B-style config for the requested A100 experiment

**Files:**
- Create: `scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json`
- Test: `scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json`

**Step 1: Write the failing test**

Validate the config fields we need:

```python
import json
cfg = json.load(open('scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json'))
assert cfg['n_embd'] == 12288
assert cfg['n_head'] == 96
assert cfg['n_layer'] == 96
assert cfg['n_ctx'] == 2048
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
python3 - <<'PY'
import json
json.load(open('scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json'))
PY
```

Expected: fail with file not found.

**Step 3: Write minimal implementation**

Create the config JSON by adapting the existing GPT config format already used in:
- `scripts/asplos/data/DLmodel_configs/gpt3_xl.json`
- `scripts/asplos/data/DLmodel_configs/gpt3_27.json`

Use the user-requested architecture values and keep the schema compatible with current tracing code.

**Step 4: Run test to verify it passes**

Run the validation snippet from Step 1.

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json
git commit -m "feat: add GPT-3 175B A100 model config"
```

### Task 5: Add minimal prefill/decode execution support without breaking inf/train

**Files:**
- Modify: `scripts/pred.py`
- Modify: `neusight/Prediction/predictor.py`
- Modify: `neusight/Tracing/trace.py`
- Modify: `neusight/Tracing/parse.py` only if decode needs explicit parse adjustments
- Create: `tests/test_prefill_decode_modes.py`

**Step 1: Write the failing test**

Create `tests/test_prefill_decode_modes.py` with behavior-level checks such as:

```python
from pathlib import Path
from neusight.Prediction.predictor import NeusightPredictor

def test_prefill_and_decode_options_are_accepted_without_changing_existing_modes():
    # pseudo-check: parser / mode normalization accepts prefill and decode
    assert 'prefill' in {'prefill', 'decode', 'inf', 'train'}
    assert 'decode' in {'prefill', 'decode', 'inf', 'train'}
```

Add one focused test per behavior:
- mode normalization accepts `prefill`
- mode normalization accepts `decode`
- legacy `inf` and `train` remain valid
- decode path uses cache-aware single-step semantics or the documented approximation hook

**Step 2: Run test to verify it fails**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
PYTHONPATH=. pytest tests/test_prefill_decode_modes.py -q
```

Expected: fail because the new mode support does not exist yet.

**Step 3: Write minimal implementation**

Implement the smallest coherent extension:
- normalize execution types in `scripts/pred.py` / `neusight/Prediction/predictor.py`
- keep existing `inf` and `train` behavior unchanged
- map `prefill` to the full-prompt forward path
- map `decode` to the cache-aware or explicitly documented minimal approximation path
- if decode cannot be represented exactly by existing tracing, encapsulate the approximation in one place and document it in code comments

**Step 4: Run test to verify it passes**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
PYTHONPATH=. pytest tests/test_prefill_decode_modes.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/pred.py neusight/Prediction/predictor.py neusight/Tracing/trace.py neusight/Tracing/parse.py tests/test_prefill_decode_modes.py
git commit -m "feat: add prefill and decode execution modes"
```

### Task 6: Generate the requested A100 GPT-3 Prefill/Decode outputs and verify them

**Files:**
- Create: `scripts/example/gpt3_a100_prefill_decode.sh`
- Create: `docs/reports/2026-03-18-a100-gpt3-prefill-decode.md`
- Modify: `scripts/asplos/results/**` or a separate result directory for the new run
- Test: generated JSON/CSV outputs for prefill and decode

**Step 1: Write the failing test**

Define required output artifacts:

```text
- one output JSON for A100 prefill
- one output JSON for A100 decode
- one report with extracted latency numbers
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
test -f scripts/asplos/results/prediction/NVIDIA_A100-PCIE-40GB/neusight/gpt3_175b_a100-prefill-128-1.json
```

Expected: fail because the files do not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/example/gpt3_a100_prefill_decode.sh` to run:
- A100 device config
- `gpt3_175b_a100.json`
- prefill with `batch_size=1`, `sequence_length=128`
- decode with the chosen decode semantics and the user’s requested `sequence_length=2048`

Then create `docs/reports/2026-03-18-a100-gpt3-prefill-decode.md` documenting:
- exact commands
- exact output file paths
- extracted latency values
- whether decode is exact or approximated relative to the paper/implementation

**Step 4: Run test to verify it passes**

Run:

```bash
cd /work/NeuSight/.worktrees/paper-repro-a100
bash scripts/example/gpt3_a100_prefill_decode.sh
python3 - <<'PY'
import json
from pathlib import Path
base = Path('scripts/asplos/results/prediction/NVIDIA_A100-PCIE-40GB/neusight')
for name in [
    'gpt3_175b_a100-prefill-128-1.json',
    'gpt3_175b_a100-decode-2048-1.json',
]:
    path = base / name
    assert path.exists(), path
    data = json.load(open(path))
    assert 'e2e_latency' in data
print('a100-prefill-decode-ok')
PY
```

Expected: `a100-prefill-decode-ok`

**Step 5: Commit**

```bash
git add scripts/example/gpt3_a100_prefill_decode.sh scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json scripts/asplos/results docs/reports/2026-03-18-a100-gpt3-prefill-decode.md
git commit -m "feat: generate A100 GPT-3 prefill and decode results"
```
