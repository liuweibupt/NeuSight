from pathlib import Path

from neusight.Analysis.a100_lpddr5x_summary import (
    DEFAULT_GEMM_CASES,
    render_markdown,
    write_summary_artifacts,
)


def _fake_summary(config_name, config_path, stage, operation_type, bw, latency_ms, throughput, unit, params):
    return {
        'row_kind': 'summary',
        'parent_stage': None,
        'component_name': None,
        'config_path': config_path,
        'config_name': config_name,
        'stage': stage,
        'operation_type': operation_type,
        'memory_bandwidth_gbps': bw,
        'latency_sec': latency_ms / 1000.0,
        'latency_ms': latency_ms,
        'tokens_per_s': throughput if unit == 'tok/s' else None,
        'throughput_per_s': throughput,
        'throughput_unit': unit,
        'tokens_per_s_per_mm2': None,
        'parameter_summary': params,
        'batch_size': 1,
        'seq_len': 128 if stage == 'prefill' else 2048 if stage == 'decode' else None,
        'd_model': 12288,
        'n_heads': 96,
        'data_type': 'fp16',
        'gemm_m': None,
        'gemm_n': None,
        'gemm_k': None,
        'compute_area_mm2': 0.0,
        'io_area_mm2': 0.0,
        'l3_area_mm2': 0.0,
        'total_area_mm2': 0.0,
        'slowdown_vs_ga100': 1.0 if config_name == 'GA100' else 1.25,
        'speedup_vs_ga100': 1.0 if config_name == 'GA100' else 0.8,
    }


def _fake_breakdown(config_name, config_path, stage, component, latency_ms, slowdown):
    return {
        'row_kind': 'operator_breakdown',
        'parent_stage': stage,
        'component_name': component,
        'config_path': config_path,
        'config_name': config_name,
        'stage': stage,
        'operation_type': f'{stage}_operator',
        'memory_bandwidth_gbps': 2048.0 if config_name == 'GA100' else 819.2,
        'latency_sec': latency_ms / 1000.0,
        'latency_ms': latency_ms,
        'tokens_per_s': None,
        'throughput_per_s': None,
        'throughput_unit': None,
        'tokens_per_s_per_mm2': None,
        'parameter_summary': stage,
        'batch_size': None,
        'seq_len': None,
        'd_model': 12288,
        'n_heads': 96,
        'data_type': 'fp16',
        'gemm_m': None,
        'gemm_n': None,
        'gemm_k': None,
        'compute_area_mm2': 0.0,
        'io_area_mm2': 0.0,
        'l3_area_mm2': 0.0,
        'total_area_mm2': 0.0,
        'slowdown_vs_ga100': slowdown,
        'speedup_vs_ga100': 1.0 / slowdown,
    }


def _fake_rows():
    rows = [
        _fake_summary('GA100', 'configs/GA100.json', 'prefill', 'transformer_prefill', 2048.0, 3.0, 42.0, 'tok/s', 'prefill params'),
        _fake_summary('A100-LPDDR5X', 'configs/A100-LPDDR5X.json', 'prefill', 'transformer_prefill', 819.2, 4.0, 31.5, 'tok/s', 'prefill params'),
        _fake_summary('GA100', 'configs/GA100.json', 'decode', 'transformer_decode', 2048.0, 1.0, 1.0, 'tok/s', 'decode params'),
        _fake_summary('A100-LPDDR5X', 'configs/A100-LPDDR5X.json', 'decode', 'transformer_decode', 819.2, 1.4, 0.71, 'tok/s', 'decode params'),
    ]
    for label, m, n, k in DEFAULT_GEMM_CASES:
        rows.append(_fake_summary('GA100', 'configs/GA100.json', label, 'matmul', 2048.0, 0.1, 10.0, 'matmul/s', f'gemm: M={m}, N={n}, K={k}, data_type=fp16'))
        rows.append(_fake_summary('A100-LPDDR5X', 'configs/A100-LPDDR5X.json', label, 'matmul', 819.2, 0.12, 8.3, 'matmul/s', f'gemm: M={m}, N={n}, K={k}, data_type=fp16'))
    for stage in ('prefill', 'decode'):
        for component in ('qkv', 'q_mul_k', 'a_mul_v', 'h_matmul0', 'h1_matmul1', 'h2_matmul2', 'softmax', 'layernorm0', 'layernorm1', 'gelu'):
            rows.append(_fake_breakdown('GA100', 'configs/GA100.json', stage, component, 0.2, 1.0))
            rows.append(_fake_breakdown('A100-LPDDR5X', 'configs/A100-LPDDR5X.json', stage, component, 0.3, 1.5))
    return rows


def test_render_markdown_mentions_prefill_decode_and_default_gemms():
    md = render_markdown(_fake_rows())
    assert 'Prefill parameters' in md
    assert 'Decode parameters' in md
    for label, _, _, _ in DEFAULT_GEMM_CASES:
        assert label in md


def test_write_summary_artifacts_creates_expected_files(tmp_path: Path):
    write_summary_artifacts(_fake_rows(), tmp_path)
    expected = [
        'a100_lpddr5x_summary.md',
        'a100_lpddr5x_summary.csv',
        'a100_lpddr5x_plot_data.csv',
        'a100_lpddr5x_throughput_bars.png',
        'a100_lpddr5x_operator_slowdown.png',
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
