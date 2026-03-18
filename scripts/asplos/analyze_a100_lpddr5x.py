import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / 'scripts' / 'asplos' / 'results'
MODEL_CONFIG = ROOT / 'scripts' / 'asplos' / 'data' / 'DLmodel_configs' / 'gpt3_175b_a100.json'
PREDICTOR_PATH = ROOT / 'scripts' / 'asplos' / 'data' / 'predictor' / 'MLP_WAVE'
TILE_DATASET_DIR = ROOT / 'scripts' / 'asplos' / 'data' / 'dataset' / 'train'
GA100_CONFIG = ROOT / 'scripts' / 'asplos' / 'data' / 'device_configs' / 'NVIDIA_GA100_HBM2_2048.json'
LPDDR5X_CONFIG = ROOT / 'scripts' / 'asplos' / 'data' / 'device_configs' / 'NVIDIA_A100_LPDDR5X_819_2.json'
OUT_DIR = RESULT_DIR / 'bandwidth_sensitivity'

from neusight.Analysis.bandwidth_sensitivity import summarize_components


MODES = {
    'prefill': {'sequence_length': 128, 'batch_size': 1, 'tokens_per_step': 128},
    'decode': {'sequence_length': 2048, 'batch_size': 1, 'tokens_per_step': 1},
}


def _device_output_dir(config_path: Path) -> str:
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg['Device'].replace(' ', '_')


def _prediction_csv(config_path: Path, mode: str) -> Path:
    spec = MODES[mode]
    model_name = MODEL_CONFIG.stem
    device_dir = _device_output_dir(config_path)
    return RESULT_DIR / 'prediction' / device_dir / 'neusight' / f"{model_name}-{mode}-{spec['sequence_length']}-{spec['batch_size']}.csv"


def run_prediction(config_path: Path, mode: str):
    spec = MODES[mode]
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'pred.py'),
        '--predictor_name', 'neusight',
        '--predictor_path', str(PREDICTOR_PATH),
        '--device_config_path', str(config_path),
        '--model_config_path', str(MODEL_CONFIG),
        '--sequence_length', str(spec['sequence_length']),
        '--batch_size', str(spec['batch_size']),
        '--execution_type', mode,
        '--tile_dataset_dir', str(TILE_DATASET_DIR),
        '--result_dir', str(RESULT_DIR),
    ]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def _format_throughput(value: float) -> str:
    if math.isinf(value):
        return 'N/A (~0ms)'
    return f'{value:.1f}'


def _format_retained(value: float) -> str:
    if math.isnan(value):
        return 'N/A'
    return f'{value:.1f}%'


def _plot_mode(ax, df: pd.DataFrame, title: str):
    labels = df['Component'].tolist()
    ga = df['GA100 throughput'].tolist()
    lp = df['A100-LPDDR5X throughput'].tolist()

    finite = [x for x in ga + lp if not math.isinf(x)]
    cap = max(finite) * 1.10 if finite else 1.0
    ga_plot = [cap if math.isinf(x) else x for x in ga]
    lp_plot = [cap if math.isinf(x) else x for x in lp]

    y = list(range(len(labels)))
    width = 0.38
    ax.barh([i - width / 2 for i in y], ga_plot, height=width, label='GA100 HBM2 2048 GB/s', color='#4C78A8')
    ax.barh([i + width / 2 for i in y], lp_plot, height=width, label='A100-LPDDR5X 819.2 GB/s', color='#F58518')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('throughput (tok/s)')
    ax.set_title(title)

    for i, retained in enumerate(df['LPDDR5X retained'].tolist()):
        x = lp_plot[i]
        txt = _format_retained(retained)
        ax.text(x * 1.01, i + width / 2, txt, va='center', fontsize=8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for mode in MODES:
        run_prediction(GA100_CONFIG, mode)
        run_prediction(LPDDR5X_CONFIG, mode)

        ga_df = pd.read_csv(_prediction_csv(GA100_CONFIG, mode))
        lp_df = pd.read_csv(_prediction_csv(LPDDR5X_CONFIG, mode))
        with open(MODEL_CONFIG) as f:
            n_layer = int(json.load(f)['n_layer'])
        summary = summarize_components(
            ga_df,
            lp_df,
            n_layer=n_layer,
            tokens_per_step=MODES[mode]['tokens_per_step'],
        )
        summary['GA100 throughput_fmt'] = summary['GA100 throughput'].map(_format_throughput)
        summary['A100-LPDDR5X throughput_fmt'] = summary['A100-LPDDR5X throughput'].map(_format_throughput)
        summary['LPDDR5X retained_fmt'] = summary['LPDDR5X retained'].map(_format_retained)
        summary.to_csv(OUT_DIR / f'{mode}_component_throughput.csv', index=False)
        summaries[mode] = summary

    prefill = summaries['prefill']
    decode = summaries['decode']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    _plot_mode(axes[0], prefill, 'Prefill component throughput')
    _plot_mode(axes[1], decode, 'Decode component throughput')
    axes[1].legend(loc='lower right')
    fig.suptitle('Bandwidth sensitivity: GA100 HBM2 vs A100-LPDDR5X', fontsize=14)
    fig.savefig(OUT_DIR / 'a100_lpddr5x_component_throughput.png', dpi=180)
    fig.savefig(OUT_DIR / 'a100_lpddr5x_component_throughput.svg')

    report_lines = [
        '# A100 LPDDR5X Bandwidth Sensitivity',
        '',
        '- This sensitivity study keeps A100 compute parameters fixed and changes only memory bandwidth.',
        '- Baseline bandwidth is set to the user-requested **2048.0 GB/s**.',
        '- Reduced bandwidth is set to **819.2 GB/s**.',
        '- Note: the repository\'s stock `NVIDIA_A100-PCIE-40GB.json` uses `Mem_Bw=1555`, but this analysis intentionally follows the user-specified 2048.0 → 819.2 what-if scenario.',
        '- In decode mode, some vector components are predicted as ~0 ms on both devices, so they are reported as `N/A (~0ms)` instead of a finite tok/s value.',
        '',
    ]
    for mode, df in [('Prefill', prefill), ('Decode', decode)]:
        report_lines.extend([
            f'## {mode}',
            '',
            '| Component | GA100 throughput | A100-LPDDR5X throughput | LPDDR5X retained |',
            '|---|---:|---:|---:|',
        ])
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['Component']} | {row['GA100 throughput_fmt']} tok/s | {row['A100-LPDDR5X throughput_fmt']} tok/s | {row['LPDDR5X retained_fmt']} |"
            )
        report_lines.append('')
    (OUT_DIR / 'a100_lpddr5x_bandwidth_report.md').write_text('\n'.join(report_lines))


if __name__ == '__main__':
    main()
