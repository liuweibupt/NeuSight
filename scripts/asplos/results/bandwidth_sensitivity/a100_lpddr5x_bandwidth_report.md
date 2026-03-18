# A100 LPDDR5X Bandwidth Sensitivity

- This sensitivity study keeps A100 compute parameters fixed and changes only memory bandwidth.
- Baseline bandwidth is set to the user-requested **2048.0 GB/s**.
- Reduced bandwidth is set to **819.2 GB/s**.
- Note: the repository's stock `NVIDIA_A100-PCIE-40GB.json` uses `Mem_Bw=1555`, but this analysis intentionally follows the user-specified 2048.0 → 819.2 what-if scenario.
- In decode mode, some vector components are predicted as ~0 ms on both devices, so they are reported as `N/A (~0ms)` instead of a finite tok/s value.

## Prefill

| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |
|---|---:|---:|---:|
| h1_matmul1 | 130.1 tok/s | 109.6 tok/s | 84.3% |
| h2_matmul2 | 117.8 tok/s | 77.3 tok/s | 65.6% |
| qkv | 176.6 tok/s | 151.2 tok/s | 85.6% |
| h_matmul0 | 476.0 tok/s | 325.2 tok/s | 68.3% |
| q_mul_k | 35911.6 tok/s | 15013.2 tok/s | 41.8% |
| a_mul_v | 35911.6 tok/s | 15013.2 tok/s | 41.8% |
| softmax | 105616.5 tok/s | 43390.4 tok/s | 41.1% |
| gelu | 33384.7 tok/s | 13177.0 tok/s | 39.5% |
| layernorm0 | 12811.8 tok/s | 5569.2 tok/s | 43.5% |
| layernorm1 | 12811.8 tok/s | 5569.2 tok/s | 43.5% |

## Decode

| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |
|---|---:|---:|---:|
| h1_matmul1 | 3.7 tok/s | 3.4 tok/s | 89.6% |
| h2_matmul2 | 3.3 tok/s | 1.8 tok/s | 54.0% |
| qkv | 5.0 tok/s | 4.5 tok/s | 89.8% |
| h_matmul0 | 13.1 tok/s | 8.2 tok/s | 62.9% |
| q_mul_k | 27.2 tok/s | 19.8 tok/s | 72.7% |
| a_mul_v | 25.8 tok/s | 24.9 tok/s | 96.5% |
| softmax | N/A (~0ms) tok/s | N/A (~0ms) tok/s | N/A |
| gelu | N/A (~0ms) tok/s | N/A (~0ms) tok/s | N/A |
| layernorm0 | N/A (~0ms) tok/s | N/A (~0ms) tok/s | N/A |
| layernorm1 | N/A (~0ms) tok/s | N/A (~0ms) tok/s | N/A |
