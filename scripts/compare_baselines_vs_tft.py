"""
scripts/compare_baselines_vs_tft.py
Generate comparison table between Phase 1 Baselines and Phase 2 TFT.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pandas as pd
from pathlib import Path

def main():
    print("Generating Baseline vs TFT Comparison...")
    
    baseline_file = Path("reports/metrics/baseline_results_summary.csv")
    if not baseline_file.exists():
        print(f"Warning: {baseline_file} not found. Ensure Phase 1 evaluation ran.")
        return
        
    df_base = pd.read_csv(baseline_file)
    
    # TFT files
    tft_stage = Path("reports/phase2/metrics/tft_stage_metrics.csv")
    tft_prog = Path("reports/phase2/metrics/tft_progression_metrics.csv")
    tft_egfr = Path("reports/phase2/metrics/tft_egfr_metrics.csv")
    
    out_dir = Path("reports/phase2/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comparisons = []
    
    # Compare Stage (Macro F1)
    if tft_stage.exists():
        try:
            df_ts = pd.read_csv(tft_stage)
            for hz in ["3m", "6m", "12m"]:
                if hz not in df_ts['horizon'].values:
                    continue
                base_subset = df_base[(df_base['target'] == 'stage') & (df_base['horizon'] == hz)]
                if not base_subset.empty:
                    best_base = base_subset.loc[base_subset['f1_macro'].idxmax()]
                    tft_val = df_ts[df_ts['horizon'] == hz]['f1_macro'].values[0]
                    diff = tft_val - best_base['f1_macro']
                    comparisons.append({
                        "Target": "Stage",
                        "Horizon": hz,
                        "Best Baseline Model": best_base['model'],
                        "Metric": "Macro F1",
                        "Baseline Score": best_base['f1_macro'],
                        "TFT Score": tft_val,
                        "Absolute Diff": diff
                    })
        except pd.errors.EmptyDataError:
            pass
                
    # Compare Progression (ROC-AUC)
    if tft_prog.exists():
        try:
            df_tp = pd.read_csv(tft_prog)
            for hz in ["3m", "6m", "12m"]:
                if hz not in df_tp['horizon'].values:
                    continue
                base_subset = df_base[(df_base['target'] == 'progression') & (df_base['horizon'] == hz)]
                if not base_subset.empty:
                    best_base = base_subset.loc[base_subset['roc_auc'].idxmax()]
                    tft_val = df_tp[df_tp['horizon'] == hz]['roc_auc'].values[0]
                    diff = tft_val - best_base['roc_auc']
                    comparisons.append({
                        "Target": "Progression",
                        "Horizon": hz,
                        "Best Baseline Model": best_base['model'],
                        "Metric": "ROC-AUC",
                        "Baseline Score": best_base['roc_auc'],
                        "TFT Score": tft_val,
                        "Absolute Diff": diff
                    })
        except pd.errors.EmptyDataError:
            pass
                
    # Compare eGFR (MAE)
    if tft_egfr.exists():
        try:
            df_te = pd.read_csv(tft_egfr)
            for hz in ["3m", "6m", "12m"]:
                if hz not in df_te['horizon'].values:
                    continue
                base_subset = df_base[(df_base['target'] == 'egfr') & (df_base['horizon'] == hz)]
                if not base_subset.empty:
                    # MAE is lower-is-better
                    best_base = base_subset.loc[base_subset['mae'].idxmin()]
                    tft_val = df_te[df_te['horizon'] == hz]['mae'].values[0]
                    diff = tft_val - best_base['mae'] # Negative is better for MAE!
                    comparisons.append({
                        "Target": "eGFR",
                        "Horizon": hz,
                        "Best Baseline Model": best_base['model'],
                        "Metric": "MAE (Lower=Better)",
                        "Baseline Score": best_base['mae'],
                        "TFT Score": tft_val,
                        "Absolute Diff": diff
                    })
        except pd.errors.EmptyDataError:
            pass
                
    df_comp = pd.DataFrame(comparisons)
    out_file = out_dir / "baseline_vs_tft_comparison.csv"
    df_comp.to_csv(out_file, index=False)
    
    print("\nComparison Summary:")
    print(df_comp.to_string())
    print(f"\nSaved comparison to {out_file}")

if __name__ == "__main__":
    main()
