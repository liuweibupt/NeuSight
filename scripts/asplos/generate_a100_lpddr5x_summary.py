from neusight.Analysis.a100_lpddr5x_summary import run_comparison, write_summary_artifacts

if __name__ == '__main__':
    rows = run_comparison()
    md_path, csv_path = write_summary_artifacts(rows)
    print(f'Markdown written to {md_path}')
    print(f'CSV written to {csv_path}')
