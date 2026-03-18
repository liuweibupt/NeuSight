import pandas as pd

from neusight.Tracing.custom_gpt import build_gpt_autoregressive_graph


def _row(df, name):
    return df.loc[df['Name'] == name].iloc[0]


def test_build_gpt_prefill_graph_uses_full_sequence_attention():
    cfg = {'n_embd': 12288, 'n_head': 96, 'n_layer': 96, 'vocab_size': 50257}
    df = build_gpt_autoregressive_graph(cfg, batch_size=1, sequence_length=128, execution_type='prefill')

    assert 'transformer_h_0_ln_1' in df['Name'].values
    assert 'add_15' in df['Name'].values
    assert _row(df, 'matmul')['FwOps'] == [('BMM', (96, 128, 128, 128))]
    assert _row(df, 'matmul_1')['FwOps'] == [('BMM', (96, 128, 128, 128))]
    assert _row(df, 'lm_head')['FwOps'] == [('Linear', (128, 12288, 50257))]


def test_build_gpt_decode_graph_uses_single_query_and_long_kv_cache():
    cfg = {'n_embd': 12288, 'n_head': 96, 'n_layer': 96, 'vocab_size': 50257}
    df = build_gpt_autoregressive_graph(cfg, batch_size=1, sequence_length=2048, execution_type='decode')

    assert _row(df, 'matmul')['FwOps'] == [('BMM', (96, 1, 128, 2048))]
    assert _row(df, 'matmul_1')['FwOps'] == [('BMM', (96, 1, 2048, 128))]
    assert _row(df, 'lm_head')['FwOps'] == [('Linear', (1, 12288, 50257))]
