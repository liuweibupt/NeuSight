import pandas as pd

from neusight.Analysis.bandwidth_sensitivity import summarize_components


def test_summarize_components_aggregates_layer_scaled_throughput_and_retention():
    base = pd.DataFrame(
        [
            {'Name': 'addmm', 'fw_latency': 2.0},
            {'Name': 'matmul', 'fw_latency': 1.0},
            {'Name': 'matmul_1', 'fw_latency': 0.5},
            {'Name': 'softmax', 'fw_latency': 0.25},
            {'Name': 'gelu', 'fw_latency': 0.125},
            {'Name': 'transformer_h_0_ln_1', 'fw_latency': 0.1},
            {'Name': 'transformer_h_0_ln_2', 'fw_latency': 0.1},
            {'Name': 'addmm_1', 'fw_latency': 1.5},
            {'Name': 'addmm_2', 'fw_latency': 3.0},
            {'Name': 'addmm_3', 'fw_latency': 4.0},
        ]
    )
    reduced = base.copy()
    reduced['fw_latency'] = reduced['fw_latency'] * 2

    out = summarize_components(base, reduced, n_layer=2, tokens_per_step=128)

    assert list(out['Component']) == [
        'h1_matmul1',
        'h2_matmul2',
        'qkv',
        'h_matmul0',
        'q_mul_k',
        'a_mul_v',
        'softmax',
        'gelu',
        'layernorm0',
        'layernorm1',
    ]

    qkv = out.loc[out['Component'] == 'qkv'].iloc[0]
    assert round(qkv['GA100 throughput'], 4) == 32000.0
    assert round(qkv['A100-LPDDR5X throughput'], 4) == 16000.0
    assert round(qkv['LPDDR5X retained'], 1) == 50.0

    ln0 = out.loc[out['Component'] == 'layernorm0'].iloc[0]
    assert round(ln0['GA100 throughput'], 4) == 640000.0
    assert round(ln0['A100-LPDDR5X throughput'], 4) == 320000.0
