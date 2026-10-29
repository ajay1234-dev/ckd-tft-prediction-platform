"""
src/data/eda.py
Exploratory Data Analysis — 16 publication-quality figures + eda_summary.json.

All figures saved to reports/figures/.
All key numeric findings saved to reports/metrics/eda_summary.json.

Rules:
  - Correct axis labels with units.
  - No truncated y-axes that exaggerate effects.
  - Log-scale axes are clearly labeled.
  - Clinical stage labels (G1–G5) used in all plots, not numeric codes.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.ckd_stage import STAGE_ORDER
from src.utils.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Consistent palette for CKD stages
_STAGE_PALETTE = {
    "G1": "#2ecc71", "G2": "#f1c40f", "G3a": "#e67e22",
    "G3b": "#e74c3c", "G4": "#8e44ad", "G5": "#2c3e50",
}
_STAGES_ORDERED = [s for s in STAGE_ORDER if s in _STAGE_PALETTE]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {path.name}")


def run_full_eda(
    train_df: pd.DataFrame,
    figures_dir: str | Path,
    metrics_dir: str | Path,
    sample_patients: int = 50,
) -> dict:
    """
    Generate all 16 EDA figures and summary JSON.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset (raw, after date parsing — before preprocessing).
    figures_dir : str or Path
    metrics_dir : str or Path
    sample_patients : int
        Number of random patients to show in trajectory plot.

    Returns
    -------
    dict : eda_summary with key numeric findings.
    """
    figures_dir = Path(figures_dir)
    metrics_dir = Path(metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # 1 ── Age distribution
    summary["age"] = _plot_age(train_df, figures_dir / "01_age_distribution.png")

    # 2 ── Sex distribution
    summary["sex"] = _plot_sex(train_df, figures_dir / "02_sex_distribution.png")

    # 3 ── BMI distribution
    summary["bmi"] = _plot_bmi(train_df, figures_dir / "03_bmi_distribution.png")

    # 4 ── CKD stage distribution
    summary["ckd_stage"] = _plot_ckd_stage(train_df, figures_dir / "04_ckd_stage_distribution.png")

    # 5 ── eGFR by stage
    summary["egfr_by_stage"] = _plot_egfr_by_stage(train_df, figures_dir / "05_egfr_by_stage.png")

    # 6 ── Serum creatinine
    summary["serum_creatinine"] = _plot_numeric(
        train_df, "serum_creatinine_mg_dl",
        "Serum Creatinine (mg/dL)", "06_serum_creatinine.png", figures_dir
    )

    # 7 ── UACR (log scale)
    summary["uacr"] = _plot_uacr(train_df, figures_dir / "07_uacr_distribution_log.png")

    # 8 ── Blood pressure
    summary["blood_pressure"] = _plot_bp(train_df, figures_dir / "08_blood_pressure.png")

    # 9 ── HbA1c
    summary["hba1c"] = _plot_numeric(
        train_df, "hba1c_pct", "HbA1c (%)", "09_hba1c_distribution.png", figures_dir
    )

    # 10 ── Hemoglobin
    summary["hemoglobin"] = _plot_numeric(
        train_df, "hemoglobin_g_dl", "Hemoglobin (g/dL)", "10_hemoglobin_distribution.png", figures_dir
    )

    # 11 ── Missing value analysis
    summary["missingness"] = _plot_missing(train_df, figures_dir / "11_missing_values.png")

    # 12 ── Visits per patient
    summary["visits_per_patient"] = _plot_visits_per_patient(
        train_df, figures_dir / "12_visits_per_patient.png"
    )

    # 13 ── eGFR trajectories
    summary["egfr_trajectory"] = _plot_egfr_trajectory(
        train_df, figures_dir / "13_egfr_trajectory.png", sample_n=sample_patients
    )

    # 14 ── CKD stage transition matrix
    summary["stage_transition"] = _plot_stage_transition(
        train_df, figures_dir / "14_stage_transition_matrix.png"
    )

    # 15 ── Diabetes & hypertension vs CKD stage
    summary["comorbidity_by_stage"] = _plot_comorbidity_by_stage(
        train_df, figures_dir / "15_comorbidity_by_stage.png"
    )

    # 16 ── Albuminuria vs CKD stage
    summary["albuminuria_by_stage"] = _plot_albuminuria_by_stage(
        train_df, figures_dir / "16_albuminuria_by_stage.png"
    )

    # 17 ── Lab correlation heatmap
    summary["correlation"] = _plot_correlation(
        train_df, figures_dir / "17_lab_correlation_heatmap.png"
    )

    # Save summary JSON
    summary_path = metrics_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"EDA summary saved → {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_age(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    age_vals = df["age"].dropna()
    ax.hist(age_vals, bins=40, color="#3498db", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Visit count")
    ax.set_title("Age Distribution of CKD Patients")
    ax.set_xlim(left=0)
    _save_fig(fig, path)
    return {"mean": round(float(age_vals.mean()), 2), "median": round(float(age_vals.median()), 2),
            "std": round(float(age_vals.std()), 2), "min": float(age_vals.min()), "max": float(age_vals.max())}


def _plot_sex(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sex"].value_counts()
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="white")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Visit count")
    ax.set_title("Sex Distribution")
    _save_fig(fig, path)
    return counts.to_dict()


def _plot_bmi(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bmi_vals = df["bmi"].dropna()
    ax.hist(bmi_vals, bins=40, color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.axvline(18.5, color="#27ae60", linestyle="--", lw=1.5, label="Underweight <18.5")
    ax.axvline(25.0, color="#f39c12", linestyle="--", lw=1.5, label="Normal <25")
    ax.axvline(30.0, color="#e74c3c", linestyle="--", lw=1.5, label="Obese ≥30")
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Visit count")
    ax.set_title("BMI Distribution")
    ax.legend(fontsize=8)
    _save_fig(fig, path)
    return {"mean": round(float(bmi_vals.mean()), 2), "std": round(float(bmi_vals.std()), 2)}


def _plot_ckd_stage(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    stage_counts = df["ckd_stage"].value_counts()
    ordered = [s for s in STAGE_ORDER if s in stage_counts.index]
    vals = [stage_counts.get(s, 0) for s in ordered]
    colors = [_STAGE_PALETTE.get(s, "#95a5a6") for s in ordered]
    bars = ax.bar(ordered, vals, color=colors, edgecolor="white")
    total = sum(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.002,
                f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("CKD Stage (KDIGO G1–G5)")
    ax.set_ylabel("Visit count")
    ax.set_title("CKD Stage Distribution")
    ax.set_ylim(top=max(vals) * 1.15)
    _save_fig(fig, path)
    return {s: {"count": int(c), "pct": round(100*c/total, 2)} for s, c in zip(ordered, vals)}


def _plot_egfr_by_stage(df, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = [s for s in STAGE_ORDER if s in df["ckd_stage"].values]
    data_by_stage = [df.loc[df["ckd_stage"] == s, "egfr_ml_min_1_73m2"].dropna().values for s in ordered]
    bp = ax.boxplot(data_by_stage, labels=ordered, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, stage in zip(bp["boxes"], ordered):
        patch.set_facecolor(_STAGE_PALETTE.get(stage, "#95a5a6"))
        patch.set_alpha(0.8)
    ax.set_xlabel("CKD Stage")
    ax.set_ylabel("eGFR (mL/min/1.73m²)")
    ax.set_title("eGFR Distribution by CKD Stage")
    ax.set_ylim(bottom=0)
    _save_fig(fig, path)
    return {s: {"median_egfr": round(float(np.median(d)), 2)} for s, d in zip(ordered, data_by_stage) if len(d) > 0}


def _plot_numeric(df, col, xlabel, filename, figures_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = df[col].dropna()
    ax.hist(vals, bins=50, color="#1abc9c", alpha=0.8, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Visit count")
    ax.set_title(f"{xlabel} Distribution")
    ax.set_xlim(left=0)
    _save_fig(fig, figures_dir / filename)
    return {"mean": round(float(vals.mean()), 3), "median": round(float(vals.median()), 3),
            "std": round(float(vals.std()), 3), "p95": round(float(vals.quantile(0.95)), 3)}


def _plot_uacr(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = df["uacr_mg_g"].dropna()
    vals_positive = vals[vals > 0]
    ax.hist(np.log10(vals_positive + 1), bins=50, color="#e67e22", alpha=0.8, edgecolor="white")
    ax.set_xlabel("UACR — log₁₀(mg/g + 1)  [log scale]")
    ax.set_ylabel("Visit count")
    ax.set_title("UACR Distribution (log scale due to right skew)")
    ax.annotate("Note: log₁₀ scale applied; original units are mg/g",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8, color="grey")
    _save_fig(fig, path)
    return {"median_mg_g": round(float(vals.median()), 2), "p95_mg_g": round(float(vals.quantile(0.95)), 2),
            "pct_above_300": round(100 * (vals > 300).mean(), 2)}


def _plot_bp(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label, color in [
        (axes[0], "systolic_bp",  "Systolic BP (mmHg)",  "#e74c3c"),
        (axes[1], "diastolic_bp", "Diastolic BP (mmHg)", "#3498db"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Visit count")
        ax.set_title(f"{label} Distribution")
        ax.set_xlim(left=0)
    fig.suptitle("Blood Pressure Distributions", fontsize=13)
    _save_fig(fig, path)
    return {
        "systolic_mean": round(float(df["systolic_bp"].mean()), 1),
        "diastolic_mean": round(float(df["diastolic_bp"].mean()), 1),
    }


def _plot_missing(df, path):
    missing_pct = df.isnull().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        # No missingness — create a simple text figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        _save_fig(fig, path)
        return {"total_missing_columns": 0}

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing_pct) * 0.3)))
    ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Missing value (%)")
    ax.set_title("Missing Value Rate by Column")
    ax.set_xlim(0, 100)
    _save_fig(fig, path)
    return {"total_missing_columns": int((missing_pct > 0).sum()),
            "max_missing_col": str(missing_pct.idxmax()),
            "max_missing_pct": round(float(missing_pct.max()), 2)}


def _plot_visits_per_patient(df, path):
    visits_pp = df.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(visits_pp.values, bins=range(1, visits_pp.max() + 2),
            color="#2980b9", alpha=0.8, edgecolor="white", align="left")
    ax.set_xlabel("Visits per patient")
    ax.set_ylabel("Number of patients")
    ax.set_title("Visits per Patient Distribution")
    _save_fig(fig, path)
    return {"min": int(visits_pp.min()), "max": int(visits_pp.max()),
            "median": float(visits_pp.median()), "mean": round(float(visits_pp.mean()), 2)}


def _plot_egfr_trajectory(df, path, sample_n=50):
    rng = np.random.default_rng(42)
    all_pids = df["patient_id"].unique()
    sample_pids = rng.choice(all_pids, size=min(sample_n, len(all_pids)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual trajectories
    for pid in sample_pids:
        sub = df[df["patient_id"] == pid].sort_values("visit_number")
        axes[0].plot(sub["visit_number"], sub["egfr_ml_min_1_73m2"],
                     alpha=0.25, lw=0.8, color="#3498db")
    axes[0].set_xlabel("Visit number")
    axes[0].set_ylabel("eGFR (mL/min/1.73m²)")
    axes[0].set_title(f"eGFR Trajectories — {sample_n} random patients")
    axes[0].set_ylim(bottom=0)

    # Right: cohort mean ± std
    cohort = df.groupby("visit_number")["egfr_ml_min_1_73m2"].agg(["mean", "std"]).reset_index()
    axes[1].plot(cohort["visit_number"], cohort["mean"], color="#e74c3c", lw=2, label="Mean eGFR")
    axes[1].fill_between(
        cohort["visit_number"],
        cohort["mean"] - cohort["std"],
        cohort["mean"] + cohort["std"],
        alpha=0.3, color="#e74c3c", label="±1 SD"
    )
    axes[1].set_xlabel("Visit number")
    axes[1].set_ylabel("eGFR (mL/min/1.73m²)")
    axes[1].set_title("Cohort eGFR Trend Over Visits")
    axes[1].legend()
    axes[1].set_ylim(bottom=0)

    fig.suptitle("eGFR Over Time — Longitudinal CKD Dataset", fontsize=13)
    _save_fig(fig, path)
    return {"cohort_egfr_visit1_mean": round(float(cohort.iloc[0]["mean"]), 2),
            "cohort_egfr_last_mean": round(float(cohort.iloc[-1]["mean"]), 2)}


def _plot_stage_transition(df, path):
    """Transition matrix: CKD stage at visit N → visit N+1."""
    df_sorted = df.sort_values(["patient_id", "visit_number"])
    df_sorted["next_stage"] = df_sorted.groupby("patient_id")["ckd_stage"].shift(-1)
    transitions = df_sorted.dropna(subset=["next_stage"])

    matrix = pd.crosstab(transitions["ckd_stage"], transitions["next_stage"])
    # Reindex to canonical stage order
    stages_present = [s for s in STAGE_ORDER if s in matrix.index or s in matrix.columns]
    matrix = matrix.reindex(index=stages_present, columns=stages_present, fill_value=0)

    # Normalize by row (transition probabilities)
    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_pct = matrix.div(row_sums, axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Transition probability (%)"})
    ax.set_xlabel("Next Stage")
    ax.set_ylabel("Current Stage")
    ax.set_title("CKD Stage Transition Probabilities (%)\n(consecutive visits)")
    _save_fig(fig, path)
    return {"dominant_transition": "see figure"}


def _plot_comorbidity_by_stage(df, path):
    stages = [s for s in STAGE_ORDER if s in df["ckd_stage"].values]
    diab_pct  = [100 * df.loc[df["ckd_stage"] == s, "diabetes"].mean()    for s in stages]
    htn_pct   = [100 * df.loc[df["ckd_stage"] == s, "hypertension"].mean() for s in stages]

    x = np.arange(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, diab_pct,  width, label="Diabetes",     color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, htn_pct,   width, label="Hypertension", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlabel("CKD Stage")
    ax.set_ylabel("Prevalence (%)")
    ax.set_title("Diabetes & Hypertension Prevalence by CKD Stage")
    ax.set_ylim(0, 100)
    ax.legend()
    _save_fig(fig, path)
    return {"stages": stages, "diabetes_pct": [round(v, 1) for v in diab_pct],
            "hypertension_pct": [round(v, 1) for v in htn_pct]}


def _plot_albuminuria_by_stage(df, path):
    if "albuminuria_category" not in df.columns:
        return {}
    ct = pd.crosstab(df["ckd_stage"], df["albuminuria_category"], normalize="index") * 100
    ordered_stages = [s for s in STAGE_ORDER if s in ct.index]
    ct = ct.reindex(ordered_stages)

    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", alpha=0.85)
    ax.set_xlabel("CKD Stage")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Albuminuria Category by CKD Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Albuminuria Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    _save_fig(fig, path)
    return {"albuminuria_categories": list(ct.columns)}


def _plot_correlation(df, path):
    lab_cols = [
        "egfr_ml_min_1_73m2", "serum_creatinine_mg_dl", "bun_mg_dl",
        "uacr_mg_g", "hba1c_pct", "hemoglobin_g_dl", "systolic_bp",
        "diastolic_bp", "fasting_glucose_mg_dl", "sodium_meq_l",
        "potassium_meq_l", "calcium_mg_dl", "phosphate_mg_dl",
        "bicarbonate_meq_l", "uric_acid_mg_dl", "bmi",
    ]
    available = [c for c in lab_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Lab & Vitals Correlation Heatmap")
    _save_fig(fig, path)
    return {"n_features": len(available), "strongest_negative": "see figure"}
