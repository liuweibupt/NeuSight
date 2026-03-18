# A100 GPT-3 Prefill/Decode Results

## Scope
This report covers the user-requested A100 GPT-3 experiment after the paper reproduction phase.

## Device
- Device config: `scripts/asplos/data/device_configs/NVIDIA_A100-PCIE-40GB.json`
- Reason for choice: this is the single-GPU A100 inference device config already present in the repository and aligned with the paper’s NVIDIA inference setup.

## Model
- Config file: `scripts/asplos/data/DLmodel_configs/gpt3_175b_a100.json`
- Parameters:
  - `n_embd = 12288`
  - `n_head = 96`
  - `n_layer = 96`
  - `n_ctx = 2048`
  - `vocab_size = 50257`

## Commands
```bash
bash scripts/example/gpt3_a100_prefill_decode.sh
```

## Semantics
- **Prefill**: custom analytical GPT forward graph with batch `1`, query length `128`, KV length `128`.
- **Decode**: custom analytical GPT forward graph with batch `1`, query length `1`, KV-cache length `2048`.

The decode path is a **minimal NeuSight extension**, not a paper-native mode. It reuses the existing predictor and aggregator, but the graph itself is synthesized analytically rather than traced from the original paper workflow.

## Output Files
- `scripts/asplos/results/prediction/NVIDIA_A100-PCIE-40GB/neusight/gpt3_175b_a100-prefill-128-1.json`
- `scripts/asplos/results/prediction/NVIDIA_A100-PCIE-40GB/neusight/gpt3_175b_a100-decode-2048-1.json`

## Results
- Prefill e2e latency: `3375.176282089218 ms`
- Decode e2e latency: `999.0218297502968 ms`

## Notes
- Both outputs are inference-only, so `fw_latency == e2e_latency` and backward-related fields are `0`.
- These numbers come from NeuSight’s learned operator predictors plus the custom prefill/decode graph construction.
