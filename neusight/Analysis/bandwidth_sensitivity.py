import pandas as pd


COMPONENT_SPECS = [
    ("h1_matmul1", "addmm_2"),
    ("h2_matmul2", "addmm_3"),
    ("qkv", "addmm"),
    ("h_matmul0", "addmm_1"),
    ("q_mul_k", "matmul"),
    ("a_mul_v", "matmul_1"),
    ("softmax", "softmax"),
    ("gelu", "gelu"),
    ("layernorm0", "transformer_h_0_ln_1"),
    ("layernorm1", "transformer_h_0_ln_2"),
]


def _component_latency_ms(df: pd.DataFrame, row_name: str, n_layer: int) -> float:
    latency = float(df.loc[df["Name"] == row_name, "fw_latency"].iloc[0])
    return latency * n_layer


def _throughput(tokens_per_step: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return float("inf")
    return tokens_per_step / (latency_ms / 1000.0)


def summarize_components(
    base_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    n_layer: int,
    tokens_per_step: float,
) -> pd.DataFrame:
    rows = []

    for component, row_name in COMPONENT_SPECS:
        base_latency = _component_latency_ms(base_df, row_name, n_layer)
        reduced_latency = _component_latency_ms(reduced_df, row_name, n_layer)
        base_tp = _throughput(tokens_per_step, base_latency)
        reduced_tp = _throughput(tokens_per_step, reduced_latency)
        retained = reduced_tp / base_tp * 100.0 if base_tp != 0 else 0.0

        rows.append(
            {
                "Component": component,
                "GA100 throughput": base_tp,
                "A100-LPDDR5X throughput": reduced_tp,
                "LPDDR5X retained": retained,
            }
        )

    return pd.DataFrame(rows)
