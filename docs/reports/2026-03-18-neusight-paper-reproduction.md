# NeuSight Paper Reproduction Report

## Inputs
- Repository: `NeuSight`
- Branch: `paper-repro-a100`
- Local ASPLOS assets used from the repository:
  - `scripts/asplos/data/predictor/MLP_WAVE`
  - `scripts/asplos/label`
  - `scripts/asplos/results`
  - `scripts/asplos/summary`
- Lightweight summary environment: `/tmp/neusight-summary-venv`
- CPU-only runtime environment for script-level reproduction: `/tmp/neusight-cpu-venv`

## Commands Run
1. Baseline path verification in the worktree.
2. Summary/table regeneration from existing repository results:
   - `cd scripts/asplos && python summarize.py && python table.py`
3. CPU-only import regression test:
   - `PYTHONPATH=. pytest tests/test_cpu_import.py -q`
4. Representative script-level prediction rerun on A100 + GPT3-XL inference:
   - `PYTHONPATH=. python scripts/pred.py --predictor_name neusight --predictor_path scripts/asplos/data/predictor/MLP_WAVE --device_config_path scripts/asplos/data/device_configs/NVIDIA_A100-PCIE-40GB.json --model_config_path scripts/asplos/data/DLmodel_configs/gpt3_xl.json --sequence_length 2048 --batch_size 2 --execution_type inf --tile_dataset_dir scripts/asplos/data/dataset/train --result_dir scripts/asplos/results`

## Regenerated Files
The reproduction workflow successfully regenerated the paper summary artifacts from local repository results:
- `scripts/asplos/summary/summary.csv`
- `scripts/asplos/summary/nvidia_inf.csv`
- `scripts/asplos/summary/nvidia_train.csv`
- `scripts/asplos/summary/perop.csv`

These regenerated files were then restored in Git to avoid noisy formatting-only diffs in the branch, but the regeneration commands completed successfully.

## A100 Results
Extracted A100 rows from the regenerated `summary.csv` (option field empty):

### NVIDIA_A100-PCIE-40GB inference
- BERT-Large, batch 8, seq 512: label `205.505536`, NeuSight `216.652422`, error `5.424129%`
- BERT-Large, batch 16, seq 512: label `396.475183`, NeuSight `414.127097`, error `4.452212%`
- GPT2-Large, batch 4, seq 1024: label `535.821106`, NeuSight `576.186991`, error `7.533463%`
- GPT2-Large, batch 8, seq 1024: label `1084.730591`, NeuSight `1179.424392`, error `8.729707%`
- GPT3-XL, batch 2, seq 2048: label `1749.461621`, NeuSight `1821.572946`, error `4.121915%`
- GPT3-XL, batch 8, seq 2048: label `7479.805371`, NeuSight `7012.610094`, error `6.246089%`
- OPT-13, batch 2, seq 2048: label `987.858740`, NeuSight `943.880685`, error `4.451857%`
- OPT-13, batch 8, seq 2048: label `3722.624414`, NeuSight `3736.610210`, error `0.375697%`
- GPT3-2.7B, batch 2, seq 2048: label `1741.581104`, NeuSight `1849.896915`, error `6.219395%`
- GPT3-2.7B, batch 8, seq 2048: label `7267.036133`, NeuSight `7168.133973`, error `1.360970%`
- SwitchTrans, batch 1, seq 512: label `223.212137`, NeuSight `262.607349`, error `17.649225%`
- SwitchTrans, batch 2, seq 512: label `403.395996`, NeuSight `449.807375`, error `11.505166%`

### NVIDIA_A100_80GB_PCIe training
- BERT-Large, batch 2, seq 512: label `213.180417`, NeuSight `213.433532`, error `0.118733%`
- BERT-Large, batch 8, seq 512: label `716.219397`, NeuSight `719.110866`, error `0.403713%`
- GPT2-Large, batch 1, seq 1024: label `524.046747`, NeuSight `496.555594`, error `5.245935%`
- GPT2-Large, batch 4, seq 1024: label `1635.742712`, NeuSight `1797.518667`, error `9.890061%`
- GPT3-XL, batch 1, seq 2048: label `2502.185364`, NeuSight `3200.329876`, error `27.901391%`
- GPT3-2.7B, batch 1, seq 2048: label `3012.924805`, NeuSight `2946.719781`, error `2.197367%`
- OPT-13, batch 1, seq 2048: label `1369.510718`, NeuSight `1499.198050`, error `9.469611%`
- OPT-13, batch 2, seq 2048: label `3188.820386`, NeuSight `3062.623569`, error `3.957476%`
- SwitchTrans, batch 1, seq 512: label `790.068829`, NeuSight `879.747146`, error `11.350697%`
- SwitchTrans, batch 2, seq 512: label `1395.078766`, NeuSight `1469.857485`, error `5.360179%`

## Delta vs Reference
- Regenerated `summary.csv` was compared against the repository’s tracked reference version for A100 rows.
- Result: A100 rows matched exactly for both:
  - `neusight_all_e2e_latency`
  - `neusight_all_e2e_err`
- Representative direct rerun also matched the tracked A100 GPT3-XL inference value:
  - rerun output: `1821.57 ms`
  - tracked summary value: `1821.572946 ms`

## Reproduction Verdict
**Result-level reproduction: successful.**

Specifically:
- the repository’s existing ASPLOS results can be re-summarized locally;
- regenerated A100 summary rows match the tracked repository reference values exactly;
- after a small CPU-only import-side-effect fix, at least one representative paper prediction command was re-executed successfully.

**Scope note:** this is a repository-local paper reproduction, not a from-scratch data-collection or full retraining reproduction. It is intentionally aligned with the user’s instruction to rely maximally on local repository content.
