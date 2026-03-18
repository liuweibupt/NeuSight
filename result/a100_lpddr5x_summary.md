# A100 vs A100-LPDDR5X Consolidated Summary

## Workload parameters

- Prefill parameters: prefill: batch_size=1, seq_len=128, d_model=12288, n_heads=96, data_type=fp16, device_count=1
- Decode parameters: decode: batch_size=1, seq_len=2048, d_model=12288, n_heads=96, data_type=fp16, device_count=1
- GEMM parameters: M=16384, N=1024, K=8192, data_type=fp16; M=16384, N=3584, K=8192, data_type=fp16; M=16384, N=8192, K=7168, data_type=fp16; M=16384, N=8192, K=1280, data_type=fp16; M=8192, N=8192, K=8192, data_type=fp16

## Summary table

| Config | Stage | Operation | BW (GB/s) | Latency (ms) | Throughput | Slowdown vs GA100 | Parameters |
|---|---|---|---:|---:|---:|---:|---|
| A100-LPDDR5X | prefill | transformer_prefill | 819.2 | 4162.495 | 30.751 tok/s | 1.338x | prefill: batch_size=1, seq_len=128, d_model=12288, n_heads=96, data_type=fp16, device_count=1 |
| GA100 | prefill | transformer_prefill | 2048.0 | 3111.046 | 41.144 tok/s | 1.000x | prefill: batch_size=1, seq_len=128, d_model=12288, n_heads=96, data_type=fp16, device_count=1 |
| A100-LPDDR5X | decode | transformer_decode | 819.2 | 1302.762 | 0.768 tok/s | 1.405x | decode: batch_size=1, seq_len=2048, d_model=12288, n_heads=96, data_type=fp16, device_count=1 |
| GA100 | decode | transformer_decode | 2048.0 | 927.444 | 1.078 tok/s | 1.000x | decode: batch_size=1, seq_len=2048, d_model=12288, n_heads=96, data_type=fp16, device_count=1 |
| A100-LPDDR5X | matmul_attention_output | matmul | 819.2 | 0.019 | 52900.096 matmul/s | 1.142x | gemm: M=16384, N=1024, K=8192, data_type=fp16 |
| GA100 | matmul_attention_output | matmul | 2048.0 | 0.017 | 60427.589 matmul/s | 1.000x | gemm: M=16384, N=1024, K=8192, data_type=fp16 |
| A100-LPDDR5X | matmul_ffn_3584x8192 | matmul | 819.2 | 0.066 | 15136.787 matmul/s | 1.154x | gemm: M=16384, N=3584, K=8192, data_type=fp16 |
| GA100 | matmul_ffn_3584x8192 | matmul | 2048.0 | 0.057 | 17461.255 matmul/s | 1.000x | gemm: M=16384, N=3584, K=8192, data_type=fp16 |
| A100-LPDDR5X | matmul_ffn_8192x7168 | matmul | 819.2 | 0.136 | 7357.081 matmul/s | 1.187x | gemm: M=16384, N=8192, K=7168, data_type=fp16 |
| GA100 | matmul_ffn_8192x7168 | matmul | 2048.0 | 0.115 | 8733.251 matmul/s | 1.000x | gemm: M=16384, N=8192, K=7168, data_type=fp16 |
| A100-LPDDR5X | matmul_qkv_projection | matmul | 819.2 | 0.024 | 41996.356 matmul/s | 1.159x | gemm: M=16384, N=8192, K=1280, data_type=fp16 |
| GA100 | matmul_qkv_projection | matmul | 2048.0 | 0.021 | 48670.347 matmul/s | 1.000x | gemm: M=16384, N=8192, K=1280, data_type=fp16 |
| A100-LPDDR5X | matmul_standard_8192 | matmul | 819.2 | 0.078 | 12889.061 matmul/s | 1.190x | gemm: M=8192, N=8192, K=8192, data_type=fp16 |
| GA100 | matmul_standard_8192 | matmul | 2048.0 | 0.065 | 15341.005 matmul/s | 1.000x | gemm: M=8192, N=8192, K=8192, data_type=fp16 |

## Prefill operator detail

| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |
|---|---:|---:|---:|
| qkv | 176.6 tok/s | 151.2 tok/s | 85.6% |
| q_mul_k | 35911.6 tok/s | 15013.2 tok/s | 41.8% |
| a_mul_v | 35911.6 tok/s | 15013.2 tok/s | 41.8% |
| h_matmul0 | 476.0 tok/s | 325.2 tok/s | 68.3% |
| h1_matmul1 | 130.1 tok/s | 109.6 tok/s | 84.3% |
| h2_matmul2 | 117.8 tok/s | 77.3 tok/s | 65.6% |
| softmax | 105616.5 tok/s | 43390.4 tok/s | 41.1% |
| layernorm0 | 12811.8 tok/s | 5569.2 tok/s | 43.5% |
| layernorm1 | 12811.8 tok/s | 5569.2 tok/s | 43.5% |
| gelu | 33384.7 tok/s | 13177.0 tok/s | 39.5% |

## Decode operator detail

| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |
|---|---:|---:|---:|
| qkv | 5.0 tok/s | 4.5 tok/s | 89.8% |
| q_mul_k | 27.2 tok/s | 19.8 tok/s | 72.7% |
| a_mul_v | 25.8 tok/s | 24.9 tok/s | 96.5% |
| h_matmul0 | 13.1 tok/s | 8.2 tok/s | 62.9% |
| h1_matmul1 | 3.7 tok/s | 3.4 tok/s | 89.6% |
| h2_matmul2 | 3.3 tok/s | 1.8 tok/s | 54.0% |
| softmax | N/A (~0ms) | N/A (~0ms) | N/A |
| layernorm0 | N/A (~0ms) | N/A (~0ms) | N/A |
| layernorm1 | N/A (~0ms) | N/A (~0ms) | N/A |
| gelu | N/A (~0ms) | N/A (~0ms) | N/A |
