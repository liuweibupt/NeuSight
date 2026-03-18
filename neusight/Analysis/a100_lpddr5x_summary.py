import csv
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from neusight.Analysis.bandwidth_sensitivity import COMPONENT_SPECS
from neusight.Prediction.predictor import MLPredictor

GPT3_D_MODEL = 12288
GPT3_N_HEADS = 96
DEFAULT_PREFILL_BATCH_SIZE = 1
DEFAULT_PREFILL_SEQ_LEN = 128
DEFAULT_DECODE_BATCH_SIZE = 1
DEFAULT_DECODE_SEQ_LEN = 2048
DEFAULT_GEMM_CASES = (
    ("matmul_qkv_projection", 16384, 8192, 1280),
    ("matmul_attention_output", 16384, 1024, 8192),
    ("matmul_ffn_8192x7168", 16384, 8192, 7168),
    ("matmul_ffn_3584x8192", 16384, 3584, 8192),
    ("matmul_standard_8192", 8192, 8192, 8192),
)
DEFAULT_CONFIGS = ("configs/GA100.json", "configs/A100-LPDDR5X.json")
DEFAULT_RESULT_DIR = Path("result")
DEFAULT_MD_PATH = DEFAULT_RESULT_DIR / "a100_lpddr5x_summary.md"
DEFAULT_CSV_PATH = DEFAULT_RESULT_DIR / "a100_lpddr5x_summary.csv"
DEFAULT_PLOT_DATA_CSV_PATH = DEFAULT_RESULT_DIR / "a100_lpddr5x_plot_data.csv"
DEFAULT_OVERVIEW_PNG = DEFAULT_RESULT_DIR / "a100_lpddr5x_throughput_bars.png"
DEFAULT_BREAKDOWN_PNG = DEFAULT_RESULT_DIR / "a100_lpddr5x_operator_throughput_breakdown.png"
DEFAULT_OPERATOR_SLOWDOWN_PNG = DEFAULT_RESULT_DIR / "a100_lpddr5x_operator_slowdown.png"
DATA_TYPE_NAME = "fp16"
BREAKDOWN_LOG_COMPONENTS = (
    "qkv",
    "q_mul_k",
    "a_mul_v",
    "h_matmul0",
    "h1_matmul1",
    "h2_matmul2",
    "softmax",
    "layernorm0",
    "layernorm1",
    "gelu",
    "allreduce_mha",
    "allreduce_ffn",
)
BREAKDOWN_COMPONENTS = tuple(
    component for component in BREAKDOWN_LOG_COMPONENTS if component not in {"allreduce_mha", "allreduce_ffn"}
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "scripts" / "asplos" / "data" / "DLmodel_configs" / "gpt3_175b_a100.json"
PREDICTOR_PATH = ROOT / "scripts" / "asplos" / "data" / "predictor" / "MLP_WAVE"
TILE_DATASET_DIR = ROOT / "scripts" / "asplos" / "data" / "dataset" / "train"
NEUSIGHT_RESULT_DIR = ROOT / "scripts" / "asplos" / "results"

MODE_SPECS = {
    "prefill": {"sequence_length": DEFAULT_PREFILL_SEQ_LEN, "batch_size": DEFAULT_PREFILL_BATCH_SIZE, "tokens_per_step": DEFAULT_PREFILL_BATCH_SIZE * DEFAULT_PREFILL_SEQ_LEN},
    "decode": {"sequence_length": DEFAULT_DECODE_SEQ_LEN, "batch_size": DEFAULT_DECODE_BATCH_SIZE, "tokens_per_step": DEFAULT_DECODE_BATCH_SIZE},
}


@lru_cache(maxsize=None)
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config_bandwidth_gbps(path: str | Path) -> float:
    config = load_config(path)
    if "memory_bandwidth_gbps" in config:
        return float(config["memory_bandwidth_gbps"])
    return float(load_config(_resolve_device_config_path(path))["Mem_Bw"])


def compute_area_breakdown_mm2(config: dict) -> dict:
    area = config.get("area", {})
    compute_area = float(area.get("compute_area_mm2", 0.0))
    io_area = float(area.get("io_area_mm2", 0.0))
    l3_area = float(area.get("l3_area_mm2", 0.0))
    return {
        "compute_area_mm2": compute_area,
        "io_area_mm2": io_area,
        "l3_area_mm2": l3_area,
        "total_area_mm2": compute_area + io_area + l3_area,
    }


def _resolve_device_config_path(path: str | Path) -> Path:
    config = load_config(path)
    device_path = config.get("neusight_device_config_path")
    if device_path is None:
        raise ValueError(f"config {path} missing neusight_device_config_path")
    return (ROOT / device_path).resolve()


def _prediction_csv(config_path: str | Path, stage: str) -> Path:
    spec = MODE_SPECS[stage]
    device = load_config(_resolve_device_config_path(config_path))["Device"].replace(" ", "_")
    return NEUSIGHT_RESULT_DIR / "prediction" / device / "neusight" / f"{MODEL_CONFIG.stem}-{stage}-{spec['sequence_length']}-{spec['batch_size']}.csv"


def _prediction_json(config_path: str | Path, stage: str) -> Path:
    return _prediction_csv(config_path, stage).with_suffix(".json")


def _run_prediction(config_path: str | Path, stage: str) -> None:
    spec = MODE_SPECS[stage]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pred.py"),
        "--predictor_name", "neusight",
        "--predictor_path", str(PREDICTOR_PATH),
        "--device_config_path", str(_resolve_device_config_path(config_path)),
        "--model_config_path", str(MODEL_CONFIG),
        "--sequence_length", str(spec["sequence_length"]),
        "--batch_size", str(spec["batch_size"]),
        "--execution_type", stage,
        "--tile_dataset_dir", str(TILE_DATASET_DIR),
        "--result_dir", str(NEUSIGHT_RESULT_DIR),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _simulate_gemm(device_config_path: str | Path, m: int, n: int, k: int) -> float:
    predictor = MLPredictor(PREDICTOR_PATH / "LINEAR", meta_table_path=TILE_DATASET_DIR / "linear.csv")
    device_cfg = load_config(device_config_path)
    return float(
        predictor.predict(opname=["linear"], kernel_arguments={"B": 1, "M": m, "N": n, "K": k}, device_config=device_cfg)
    )


def _format_transformer_summary(stage: str, batch_size: int, seq_len: int) -> str:
    return (
        f"{stage}: batch_size={batch_size}, seq_len={seq_len}, d_model={GPT3_D_MODEL}, "
        f"n_heads={GPT3_N_HEADS}, data_type={DATA_TYPE_NAME}, device_count=1"
    )


def _format_gemm_summary(m: int, n: int, k: int) -> str:
    return f"gemm: M={m}, N={n}, K={k}, data_type={DATA_TYPE_NAME}"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-30:
        return 1.0 if abs(numerator) < 1e-30 else float("inf")
    return numerator / denominator


def _throughput(value_count: float, latency_sec: float) -> float:
    if latency_sec <= 0:
        return float("inf")
    return value_count / latency_sec


def _build_summary_row(*, config_path: str, config_name: str, area: dict, bandwidth_gbps: float, stage: str, operation_type: str, latency_sec: float, throughput_per_s: float, throughput_unit: str, parameter_summary: str, batch_size: int | None = None, seq_len: int | None = None, d_model: int | None = None, n_heads: int | None = None, gemm_m: int | None = None, gemm_n: int | None = None, gemm_k: int | None = None) -> dict:
    return {
        "row_kind": "summary",
        "parent_stage": None,
        "component_name": None,
        "config_path": config_path,
        "config_name": config_name,
        "stage": stage,
        "operation_type": operation_type,
        "memory_bandwidth_gbps": bandwidth_gbps,
        "latency_sec": latency_sec,
        "latency_ms": latency_sec * 1000.0,
        "tokens_per_s": throughput_per_s if throughput_unit == "tok/s" else None,
        "throughput_per_s": throughput_per_s,
        "throughput_unit": throughput_unit,
        "tokens_per_s_per_mm2": throughput_per_s / area["total_area_mm2"] if area["total_area_mm2"] > 0 else None,
        "parameter_summary": parameter_summary,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "data_type": DATA_TYPE_NAME,
        "gemm_m": gemm_m,
        "gemm_n": gemm_n,
        "gemm_k": gemm_k,
        **area,
    }


def _build_breakdown_row(*, config_path: str, config_name: str, area: dict, bandwidth_gbps: float, parent_stage: str, component_name: str, latency_sec: float, parameter_summary: str) -> dict:
    return {
        "row_kind": "operator_breakdown",
        "parent_stage": parent_stage,
        "component_name": component_name,
        "config_path": config_path,
        "config_name": config_name,
        "stage": parent_stage,
        "operation_type": f"{parent_stage}_operator",
        "memory_bandwidth_gbps": bandwidth_gbps,
        "latency_sec": latency_sec,
        "latency_ms": latency_sec * 1000.0,
        "tokens_per_s": None,
        "throughput_per_s": None,
        "throughput_unit": None,
        "tokens_per_s_per_mm2": None,
        "parameter_summary": parameter_summary,
        "batch_size": None,
        "seq_len": None,
        "d_model": GPT3_D_MODEL,
        "n_heads": GPT3_N_HEADS,
        "data_type": DATA_TYPE_NAME,
        "gemm_m": None,
        "gemm_n": None,
        "gemm_k": None,
        **area,
    }


def _component_breakdown_rows(config_path: str | Path, stage: str, area: dict, bandwidth_gbps: float, config_name: str) -> list[dict]:
    csv_path = _prediction_csv(config_path, stage)
    df = pd.read_csv(csv_path)
    n_layer = int(load_config(MODEL_CONFIG)["n_layer"])
    rows = []
    for component_name, row_name in COMPONENT_SPECS:
        latency_ms = float(df.loc[df["Name"] == row_name, "fw_latency"].iloc[0]) * n_layer
        rows.append(
            _build_breakdown_row(
                config_path=str(config_path),
                config_name=config_name,
                area=area,
                bandwidth_gbps=bandwidth_gbps,
                parent_stage=stage,
                component_name=component_name,
                latency_sec=latency_ms / 1000.0,
                parameter_summary=_format_transformer_summary(stage, MODE_SPECS[stage]["batch_size"], MODE_SPECS[stage]["sequence_length"]),
            )
        )
    return rows


@lru_cache(maxsize=None)
def _run_comparison_cached(config_paths: tuple[str, ...], prefill_batch_size: int, prefill_seq_len: int, decode_batch_size: int, decode_seq_len: int, gemm_m: int | None, gemm_n: int | None, gemm_k: int | None) -> tuple[dict, ...]:
    rows: list[dict] = []
    gemm_cases = DEFAULT_GEMM_CASES if gemm_m is None or gemm_n is None or gemm_k is None else ((f"matmul_{gemm_m}x{gemm_n}x{gemm_k}", gemm_m, gemm_n, gemm_k),)

    for config_path in config_paths:
        config = load_config(config_path)
        config_name = str(config["name"])
        area = compute_area_breakdown_mm2(config)
        bandwidth_gbps = load_config_bandwidth_gbps(config_path)
        device_config_path = _resolve_device_config_path(config_path)

        for stage in ("prefill", "decode"):
            _run_prediction(config_path, stage)
            pred_json = load_config(_prediction_json(config_path, stage))
            parameter_summary = _format_transformer_summary(stage, MODE_SPECS[stage]["batch_size"], MODE_SPECS[stage]["sequence_length"])
            throughput = _throughput(MODE_SPECS[stage]["tokens_per_step"], float(pred_json["e2e_latency"]) / 1000.0)
            rows.append(
                _build_summary_row(
                    config_path=str(config_path),
                    config_name=config_name,
                    area=area,
                    bandwidth_gbps=bandwidth_gbps,
                    stage=stage,
                    operation_type=f"transformer_{stage}",
                    latency_sec=float(pred_json["e2e_latency"]) / 1000.0,
                    throughput_per_s=throughput,
                    throughput_unit="tok/s",
                    parameter_summary=parameter_summary,
                    batch_size=MODE_SPECS[stage]["batch_size"],
                    seq_len=MODE_SPECS[stage]["sequence_length"],
                    d_model=GPT3_D_MODEL,
                    n_heads=GPT3_N_HEADS,
                )
            )
            rows.extend(_component_breakdown_rows(config_path, stage, area, bandwidth_gbps, config_name))

        for gemm_label, case_m, case_n, case_k in gemm_cases:
            gemm_latency_ms = _simulate_gemm(device_config_path, case_m, case_n, case_k)
            rows.append(
                _build_summary_row(
                    config_path=str(config_path),
                    config_name=config_name,
                    area=area,
                    bandwidth_gbps=bandwidth_gbps,
                    stage=gemm_label,
                    operation_type="matmul",
                    latency_sec=gemm_latency_ms / 1000.0,
                    throughput_per_s=_throughput(1.0, gemm_latency_ms / 1000.0),
                    throughput_unit="matmul/s",
                    parameter_summary=_format_gemm_summary(case_m, case_n, case_k),
                    gemm_m=case_m,
                    gemm_n=case_n,
                    gemm_k=case_k,
                )
            )

    baseline_latency: dict[tuple, float] = {}
    baseline_name = load_config(DEFAULT_CONFIGS[0])["name"]
    for row in rows:
        if row["config_name"] != baseline_name:
            continue
        if row["row_kind"] == "summary":
            baseline_latency[("summary", row["stage"])] = row["latency_sec"]
        else:
            baseline_latency[("operator_breakdown", row["parent_stage"], row["component_name"])] = row["latency_sec"]

    finalized_rows = []
    for row in rows:
        if row["row_kind"] == "summary":
            baseline = baseline_latency[("summary", row["stage"])]
        else:
            baseline = baseline_latency[("operator_breakdown", row["parent_stage"], row["component_name"])]
        finalized_rows.append({**row, "slowdown_vs_ga100": _safe_ratio(row["latency_sec"], baseline), "speedup_vs_ga100": _safe_ratio(baseline, row["latency_sec"])})
    return tuple(finalized_rows)


def run_comparison(config_paths: Iterable[str | Path] = DEFAULT_CONFIGS, *, prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE, prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN, decode_batch_size: int = DEFAULT_DECODE_BATCH_SIZE, decode_seq_len: int = DEFAULT_DECODE_SEQ_LEN, gemm_m: int | None = None, gemm_n: int | None = None, gemm_k: int | None = None) -> list[dict]:
    normalized_paths = tuple(str(path) for path in config_paths)
    return [dict(row) for row in _run_comparison_cached(normalized_paths, prefill_batch_size, prefill_seq_len, decode_batch_size, decode_seq_len, gemm_m, gemm_n, gemm_k)]


def _stage_sort_key(stage: str) -> tuple[int, str]:
    if stage == "prefill":
        return (0, stage)
    if stage == "decode":
        return (1, stage)
    return (2, stage)


def _paired_breakdown(rows: list[dict], parent_stage: str) -> list[dict]:
    breakdown_rows = [row for row in rows if row["row_kind"] == "operator_breakdown" and row["parent_stage"] == parent_stage]
    pairs = []
    ga100_name = load_config(DEFAULT_CONFIGS[0])["name"]
    lp_name = load_config(DEFAULT_CONFIGS[1])["name"]
    for component_name in BREAKDOWN_COMPONENTS:
        ga100_row = next(row for row in breakdown_rows if row["component_name"] == component_name and row["config_name"] == ga100_name)
        lpddr_row = next(row for row in breakdown_rows if row["component_name"] == component_name and row["config_name"] == lp_name)
        pairs.append({"parent_stage": parent_stage, "component_name": component_name, "ga100_latency_ms": ga100_row["latency_ms"], "lpddr_latency_ms": lpddr_row["latency_ms"], "slowdown_vs_ga100": lpddr_row["slowdown_vs_ga100"]})
    return pairs


def _stage_token_count(stage: str, summary_rows: list[dict]) -> float:
    summary = next(row for row in summary_rows if row["row_kind"] == "summary" and row["stage"] == stage)
    if stage == "prefill":
        return float(summary["batch_size"] * summary["seq_len"])
    if stage == "decode":
        return float(summary["batch_size"])
    return 1.0


def _display_label(stage: str) -> str:
    mapping = {
        "prefill": "Prefill",
        "decode": "Decode",
        "matmul_qkv_projection": "QKV proj",
        "matmul_attention_output": "Attn out",
        "matmul_ffn_8192x7168": "FFN 8192x7168",
        "matmul_ffn_3584x8192": "FFN 3584x8192",
        "matmul_standard_8192": "Std 8192",
    }
    return mapping.get(stage, stage)


def _display_component_label(component: str) -> str:
    mapping = {
        "qkv": "QKV",
        "q_mul_k": "Q×K",
        "a_mul_v": "A×V",
        "h_matmul0": "O proj",
        "h1_matmul1": "FFN up",
        "h2_matmul2": "FFN down",
        "softmax": "Softmax",
        "layernorm0": "LN0",
        "layernorm1": "LN1",
        "gelu": "GeLU",
    }
    return mapping.get(component, component)


def _format_value(value: float, unit: str) -> str:
    if math.isinf(value):
        return "N/A (~0ms)"
    return f"{value:.1f} {unit}"


def render_markdown(rows: list[dict]) -> str:
    summary_rows = sorted([row for row in rows if row["row_kind"] == "summary"], key=lambda row: (_stage_sort_key(row["stage"]), row["config_name"]))
    prefill_row = next(row for row in summary_rows if row["stage"] == "prefill" and row["config_name"] == load_config(DEFAULT_CONFIGS[0])["name"])
    decode_row = next(row for row in summary_rows if row["stage"] == "decode" and row["config_name"] == load_config(DEFAULT_CONFIGS[0])["name"])
    gemm_rows = [row for row in summary_rows if row["operation_type"] == "matmul" and row["config_name"] == load_config(DEFAULT_CONFIGS[0])["name"]]
    prefill_pairs = _paired_breakdown(rows, "prefill")
    decode_pairs = _paired_breakdown(rows, "decode")

    lines = [
        "# A100 vs A100-LPDDR5X Consolidated Summary",
        "",
        "## Workload parameters",
        "",
        f"- Prefill parameters: {prefill_row['parameter_summary']}",
        f"- Decode parameters: {decode_row['parameter_summary']}",
        f"- GEMM parameters: {'; '.join(row['parameter_summary'].replace('gemm: ', '') for row in gemm_rows)}",
        "",
        "## Summary table",
        "",
        "| Config | Stage | Operation | BW (GB/s) | Latency (ms) | Throughput | Slowdown vs GA100 | Parameters |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['config_name']} | {row['stage']} | {row['operation_type']} | {row['memory_bandwidth_gbps']:.1f} | {row['latency_ms']:.3f} | {row['throughput_per_s']:.3f} {row['throughput_unit']} | {row['slowdown_vs_ga100']:.3f}x | {row['parameter_summary']} |")

    for title, pairs in (("Prefill", prefill_pairs), ("Decode", decode_pairs)):
        lines.extend(["", f"## {title} operator detail", "", "| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |", "|---|---:|---:|---:|"])
        for pair in pairs:
            token_count = _stage_token_count(title.lower(), summary_rows)
            ga100_tp = _throughput(token_count, pair['ga100_latency_ms'] / 1000.0)
            lpddr_tp = _throughput(token_count, pair['lpddr_latency_ms'] / 1000.0)
            retained = 100.0 / pair['slowdown_vs_ga100'] if not math.isinf(pair['slowdown_vs_ga100']) else float('nan')
            if math.isinf(ga100_tp) or math.isinf(lpddr_tp):
                ga_txt = 'N/A (~0ms)'
                lp_txt = 'N/A (~0ms)'
                ret_txt = 'N/A'
            else:
                ga_txt = f"{ga100_tp:.1f} tok/s"
                lp_txt = f"{lpddr_tp:.1f} tok/s"
                ret_txt = f"{retained:.1f}%"
            lines.append(f"| {pair['component_name']} | {ga_txt} | {lp_txt} | {ret_txt} |")
    lines.append("")
    return "\n".join(lines)


def write_summary_csv(path: str | Path = DEFAULT_CSV_PATH, rows: list[dict] | None = None) -> Path:
    if rows is None:
        rows = run_comparison()
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _apply_publication_style() -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save_figure(fig, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    return output_path


def _paired_summary_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    summary_rows = [row for row in rows if row["row_kind"] == "summary"]
    ga_name = load_config(DEFAULT_CONFIGS[0])["name"]
    lp_name = load_config(DEFAULT_CONFIGS[1])["name"]
    pairs = []
    for stage in sorted({row['stage'] for row in summary_rows}, key=_stage_sort_key):
        ga = next(row for row in summary_rows if row['stage'] == stage and row['config_name'] == ga_name)
        lp = next(row for row in summary_rows if row['stage'] == stage and row['config_name'] == lp_name)
        pairs.append((ga, lp))
    return pairs


def _normalize_pair(left: float, right: float) -> tuple[float, float]:
    if math.isinf(left) and math.isinf(right):
        return 1.0, 1.0
    finite = [v for v in (left, right) if not math.isinf(v)]
    denom = max(finite) if finite else 1.0
    return (1.0 if math.isinf(left) else left / denom, 1.0 if math.isinf(right) else right / denom)


def _lpddr_percent(left: float, right: float) -> float:
    if left <= 0 or math.isinf(left) or math.isinf(right):
        return float('nan')
    return right / left * 100.0


def _plot_slowdown_overview(rows: list[dict], path: str | Path = DEFAULT_OVERVIEW_PNG) -> Path:
    _apply_publication_style()
    pairs = _paired_summary_rows(rows)
    transformer_pairs = [pair for pair in pairs if pair[0]['throughput_unit'] == 'tok/s']
    gemm_pairs = [pair for pair in pairs if pair[0]['throughput_unit'] != 'tok/s']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    colors = {'HBM': '#4C78A8', 'LPDDR5X': '#E45756'}
    width = 0.36
    for ax, panel_pairs, title in ((axes[0], transformer_pairs, 'Transformer throughput'), (axes[1], gemm_pairs, 'GEMM throughput')):
        labels = [_display_label(pair[0]['stage']) for pair in panel_pairs]
        x = list(range(len(labels)))
        raw = [(pair[0]['throughput_per_s'], pair[1]['throughput_per_s']) for pair in panel_pairs]
        normalized = [_normalize_pair(a, b) for a, b in raw]
        hbm_values = [a for a, _ in normalized]
        lp_values = [b for _, b in normalized]
        ax.bar([i - width/2 for i in x], hbm_values, width=width, color=colors['HBM'], label='GA100')
        ax.bar([i + width/2 for i in x], lp_values, width=width, color=colors['LPDDR5X'], label='A100-LPDDR5X')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Normalized throughput')
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        for pos, (left, right), leftn, rightn in zip(x, raw, hbm_values, lp_values):
            txt = 'N/A' if math.isnan(_lpddr_percent(left, right)) else f"{_lpddr_percent(left, right):.1f}%"
            ax.text(pos + width/2, max(rightn*0.55, 0.08), txt, ha='center', va='center', fontsize=8)
    axes[1].legend(loc='upper right')
    return _save_figure(fig, path)


def _plot_operator_breakdown(rows: list[dict], path: str | Path = DEFAULT_BREAKDOWN_PNG) -> Path:
    _apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    width = 0.36
    for ax, stage in zip(axes, ('prefill', 'decode')):
        pairs = _paired_breakdown(rows, stage)
        labels = [_display_component_label(pair['component_name']) for pair in pairs]
        x = list(range(len(labels)))
        summary_rows = [row for row in rows if row['row_kind'] == 'summary']
        token_count = _stage_token_count(stage, summary_rows)
        raw = [(_throughput(token_count, pair['ga100_latency_ms'] / 1000.0), _throughput(token_count, pair['lpddr_latency_ms'] / 1000.0)) for pair in pairs]
        normalized = [_normalize_pair(a, b) for a, b in raw]
        ax.bar([i - width/2 for i in x], [a for a, _ in normalized], width=width, color='#4C78A8')
        ax.bar([i + width/2 for i in x], [b for _, b in normalized], width=width, color='#E45756')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha='right')
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Normalized throughput')
        ax.set_title(f'{stage.capitalize()} operator throughput breakdown')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    return _save_figure(fig, path)


def _plot_operator_slowdown(rows: list[dict], path: str | Path = DEFAULT_OPERATOR_SLOWDOWN_PNG) -> Path:
    _apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    colors = plt.get_cmap('tab20')
    for ax, stage in zip(axes, ('prefill', 'decode')):
        pairs = _paired_breakdown(rows, stage)
        labels = [_display_component_label(pair['component_name']) for pair in pairs]
        x = list(range(len(labels)))
        vals = [pair['slowdown_vs_ga100'] for pair in pairs]
        ax.bar(x, vals, color=[colors(i % 20) for i in range(len(labels))])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha='right')
        ax.set_ylabel('Slowdown vs GA100')
        ax.set_title(f'{stage.capitalize()} operator slowdown')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    return _save_figure(fig, path)


def write_summary_markdown(rows: list[dict], path: str | Path = DEFAULT_MD_PATH) -> Path:
    md_path = Path(path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(rows), encoding='utf-8')
    return md_path


def write_plot_data_csv(rows: list[dict], path: str | Path = DEFAULT_PLOT_DATA_CSV_PATH) -> Path:
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [row for row in rows if row['row_kind'] == 'summary']
    fieldnames = ['plot_kind','panel','stage','component_name','config_name','config_label','value','normalized_value','retained_percent_vs_hbm','value_unit','slowdown_vs_ga100','memory_bandwidth_gbps']
    with plot_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ga, lp in _paired_summary_rows(rows):
            panel = 'transformer_throughput' if ga['throughput_unit'] == 'tok/s' else 'gemm_throughput'
            ga_n, lp_n = _normalize_pair(ga['throughput_per_s'], lp['throughput_per_s'])
            writer.writerow({'plot_kind':'throughput_pair','panel':panel,'stage':ga['stage'],'component_name':'','config_name':ga['config_name'],'config_label':'HBM','value':ga['throughput_per_s'],'normalized_value':ga_n,'retained_percent_vs_hbm':100.0,'value_unit':ga['throughput_unit'],'slowdown_vs_ga100':ga['slowdown_vs_ga100'],'memory_bandwidth_gbps':ga['memory_bandwidth_gbps']})
            writer.writerow({'plot_kind':'throughput_pair','panel':panel,'stage':lp['stage'],'component_name':'','config_name':lp['config_name'],'config_label':'LPDDR5X','value':lp['throughput_per_s'],'normalized_value':lp_n,'retained_percent_vs_hbm':_lpddr_percent(ga['throughput_per_s'], lp['throughput_per_s']),'value_unit':lp['throughput_unit'],'slowdown_vs_ga100':lp['slowdown_vs_ga100'],'memory_bandwidth_gbps':lp['memory_bandwidth_gbps']})
        for stage in ('prefill','decode'):
            token_count = _stage_token_count(stage, summary_rows)
            for pair in _paired_breakdown(rows, stage):
                ga = _throughput(token_count, pair['ga100_latency_ms']/1000.0)
                lp = _throughput(token_count, pair['lpddr_latency_ms']/1000.0)
                ga_n, lp_n = _normalize_pair(ga, lp)
                writer.writerow({'plot_kind':'operator_throughput_pair','panel':stage,'stage':stage,'component_name':pair['component_name'],'config_name':'GA100','config_label':'HBM','value':ga,'normalized_value':ga_n,'retained_percent_vs_hbm':100.0,'value_unit':'tok/s','slowdown_vs_ga100':1.0,'memory_bandwidth_gbps':load_config_bandwidth_gbps(DEFAULT_CONFIGS[0])})
                writer.writerow({'plot_kind':'operator_throughput_pair','panel':stage,'stage':stage,'component_name':pair['component_name'],'config_name':'A100-LPDDR5X','config_label':'LPDDR5X','value':lp,'normalized_value':lp_n,'retained_percent_vs_hbm':_lpddr_percent(ga, lp),'value_unit':'tok/s','slowdown_vs_ga100':pair['slowdown_vs_ga100'],'memory_bandwidth_gbps':load_config_bandwidth_gbps(DEFAULT_CONFIGS[1])})
                writer.writerow({'plot_kind':'operator_slowdown','panel':stage,'stage':stage,'component_name':pair['component_name'],'config_name':'A100-LPDDR5X','config_label':'LPDDR5X','value':pair['slowdown_vs_ga100'],'normalized_value':pair['slowdown_vs_ga100'],'retained_percent_vs_hbm':(100.0 / pair['slowdown_vs_ga100']) if pair['slowdown_vs_ga100'] else None,'value_unit':'ratio','slowdown_vs_ga100':pair['slowdown_vs_ga100'],'memory_bandwidth_gbps':load_config_bandwidth_gbps(DEFAULT_CONFIGS[1])})
    return plot_path


def write_summary_artifacts(rows: list[dict], output_dir: str | Path = DEFAULT_RESULT_DIR) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    _plot_slowdown_overview(rows, output_dir / DEFAULT_OVERVIEW_PNG.name)
    _plot_operator_breakdown(rows, output_dir / DEFAULT_BREAKDOWN_PNG.name)
    _plot_operator_slowdown(rows, output_dir / DEFAULT_OPERATOR_SLOWDOWN_PNG.name)
    write_plot_data_csv(rows, output_dir / DEFAULT_PLOT_DATA_CSV_PATH.name)
    md_path = write_summary_markdown(rows, output_dir / DEFAULT_MD_PATH.name)
    csv_path = write_summary_csv(output_dir / DEFAULT_CSV_PATH.name, rows)
    return md_path, csv_path


if __name__ == '__main__':
    result_rows = run_comparison()
    md_path, csv_path = write_summary_artifacts(result_rows)
    print(f'Markdown written to {md_path}')
    print(f'CSV written to {csv_path}')
