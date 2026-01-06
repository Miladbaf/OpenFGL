"""
ihsan/make_graphfl_tables_from_npy.py

Creates 3 comparison tables from OpenFGL Graph-FL experiment .npy outputs:

Table 1) Baselines (ours) vs OpenFGL paper Table 6 (Graph-FL) + AIDS FedAvg reference.
Table 2) FedAvg vs FedALA vs FedALA-R at K=10 (client size fixed).
Table 3) Scalability: effect of client size K on FedAvg/FedALA/FedALA-R.

Robust to:
- numpy pickle compat issues (numpy.core vs numpy._core rename)
- different accuracy field names:
    * best_test_accuracy              (often fraction in [0,1])
    * best_test_accuracy_percent      (already in percent)

Outputs:
  ihsan/outputs_tables/
    table1_baselines_vs_paper_focus.csv / .tex
    table2_k10_fedavg_fedala_fedalar.csv / .tex
    table3_scalability_clients.csv / .tex
    graphfl_tables_payload.npy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# NumPy pickle compat patch (numpy._core <-> numpy.core)
# -----------------------------------------------------------------------------
def _patch_numpy_core_aliases() -> None:
    try:
        import numpy._core  # noqa: F401
        return
    except Exception:
        pass

    try:
        import numpy.core as core  # type: ignore
    except Exception:
        return

    sys.modules.setdefault("numpy._core", core)
    for sub in ("_multiarray_umath", "multiarray", "umath"):
        try:
            sys.modules.setdefault(f"numpy._core.{sub}", getattr(core, sub))
        except Exception:
            continue


_patch_numpy_core_aliases()


# -----------------------------------------------------------------------------
# Paths (repo-relative)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

BASELINE_NPY = REPO_ROOT / "ihsan" / "results_graphfl_table6_baselines2.npy"
FEDALA_NPY = REPO_ROOT / "ihsan" / "graphfl_fedala_fedalar_results.npy"
SCALABILITY_NPY = REPO_ROOT / "ihsan" / "results_scalability_graphfl_clients.npy"

OUT_DIR = REPO_ROOT / "ihsan" / "outputs_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Paper references (Graph-FL Table 6, accuracy in %)
# -----------------------------------------------------------------------------
PAPER_TABLE6: Dict[str, Dict[str, Tuple[float, float]]] = {
    "fedavg": {
        "MUTAG": (78.9, 2.9),
        "BZR": (76.5, 1.3),
        "COX2": (79.0, 1.7),
        "PROTEINS": (80.1, 1.5),
    },
    "fedprox": {
        "MUTAG": (76.5, 2.4),
        "BZR": (81.8, 1.7),
        "COX2": (77.2, 1.6),
        "PROTEINS": (77.4, 1.7),
    },
    "scaffold": {
        "MUTAG": (75.4, 2.9),
        "BZR": (82.3, 1.8),
        "COX2": (82.0, 1.4),
        "PROTEINS": (79.9, 1.1),
    },
    "gcfl_plus": {
        "MUTAG": (82.6, 2.6),
        "BZR": (87.8, 1.9),
        "COX2": (82.6, 2.3),
        "PROTEINS": (83.6, 1.3),
    },
    "fedstar": {
        "MUTAG": (84.7, 2.6),
        "BZR": (89.1, 1.5),
        "COX2": (80.6, 2.3),
        "PROTEINS": (84.5, 1.7),
    },
}

# AIDS FedAvg reference you provided (not in Table 6)
PAPER_AIDS_FEDAVG = (94.2, 0.7)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    payload = np.load(str(path), allow_pickle=True).item()
    if not isinstance(payload, dict) or "runs" not in payload:
        raise ValueError(f"Unexpected npy payload format in {path}")
    return payload


def to_runs_df(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload.get("runs", []))


# -----------------------------------------------------------------------------
# Normalization + summarization
# -----------------------------------------------------------------------------
def add_accuracy_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds column 'acc_percent' in [%].
    - If 'best_test_accuracy_percent' exists, use it directly.
    - Else use 'best_test_accuracy' and convert fractions to percent.
    """
    if df.empty:
        df["acc_percent"] = []
        return df

    if "best_test_accuracy_percent" in df.columns:
        df["acc_percent"] = pd.to_numeric(df["best_test_accuracy_percent"], errors="coerce")
        return df

    if "best_test_accuracy" not in df.columns:
        raise KeyError(
            "Runs do not contain 'best_test_accuracy' or 'best_test_accuracy_percent'. "
            f"Columns: {list(df.columns)}"
        )

    s = pd.to_numeric(df["best_test_accuracy"], errors="coerce")
    mx = float(s.dropna().max()) if not s.dropna().empty else 0.0
    df["acc_percent"] = s * 100.0 if mx <= 1.2 else s
    return df


def summarize_runs(df: pd.DataFrame, group_cols: List[str], value_col: str = "acc_percent") -> pd.DataFrame:
    """
    mean/std/n over groups, ignoring NaNs.
    IMPORTANT: std uses ddof=0 to match your printed summaries.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "std", "n"])

    if value_col not in df.columns:
        raise KeyError(f"Missing '{value_col}' in dataframe columns: {list(df.columns)}")

    g = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(
            mean="mean",
            std=lambda x: float(x.std(ddof=0)) if len(x.dropna()) > 1 else 0.0,
            n="count",
        )
        .reset_index()
    )
    return g


def fmt_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------
def build_table1_baselines_vs_paper(baseline_runs: pd.DataFrame) -> pd.DataFrame:
    s = summarize_runs(baseline_runs, group_cols=["method", "dataset"])
    rows = []

    for _, r in s.iterrows():
        method = str(r["method"])
        dataset = str(r["dataset"])
        ours_mean = float(r["mean"])
        ours_std = float(r["std"])

        paper_mean = float("nan")
        paper_std = float("nan")
        if dataset == "AIDS" and method == "fedavg":
            paper_mean, paper_std = PAPER_AIDS_FEDAVG
        elif method in PAPER_TABLE6 and dataset in PAPER_TABLE6[method]:
            paper_mean, paper_std = PAPER_TABLE6[method][dataset]

        delta = ours_mean - paper_mean if not pd.isna(paper_mean) else float("nan")

        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "paper_mean±std": fmt_mean_std(paper_mean, paper_std) if not pd.isna(paper_mean) else "-",
                "ours_mean±std": fmt_mean_std(ours_mean, ours_std),
                "delta_mean(ours-paper)": (f"{delta:+.2f}" if not pd.isna(delta) else "-"),
                "ours_n": int(r["n"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["method", "dataset"]).reset_index(drop=True)


def build_table2_k10_fedavg_fedala_fedalar(
    baseline_runs: pd.DataFrame,
    fedala_runs: pd.DataFrame,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    base = baseline_runs[baseline_runs["method"].isin(["fedavg"])].copy()
    ala = fedala_runs[fedala_runs["method"].isin(["fedala", "fedala_r"])].copy()

    df = pd.concat([base, ala], ignore_index=True)
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()

    s = summarize_runs(df, group_cols=["method", "dataset"])
    s["cell"] = s.apply(lambda r: fmt_mean_std(float(r["mean"]), float(r["std"])), axis=1)

    pivot = (
        s.pivot(index="method", columns="dataset", values="cell")
        .reindex(index=["fedavg", "fedala", "fedala_r"])
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def build_table3_scalability(scal_runs: pd.DataFrame, datasets: Optional[List[str]] = None) -> pd.DataFrame:
    df = scal_runs.copy()
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()

    s = summarize_runs(df, group_cols=["method", "dataset", "K"])
    s["cell"] = s.apply(lambda r: fmt_mean_std(float(r["mean"]), float(r["std"])), axis=1)

    pivot = s.pivot(index=["method", "dataset"], columns="K", values="cell").sort_index().reset_index()
    pivot.columns.name = None

    # Rename numeric K columns to "K=5" etc.
    new_cols = []
    for c in pivot.columns:
        if isinstance(c, (int, float)) and str(c).replace(".", "", 1).isdigit():
            new_cols.append(f"K={int(c)}")
        else:
            new_cols.append(c)
    pivot.columns = new_cols
    return pivot


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------
def save_table(df: pd.DataFrame, stem: str) -> None:
    csv_path = OUT_DIR / f"{stem}.csv"
    tex_path = OUT_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"[saved] {csv_path}")
    print(f"[saved] {tex_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    baseline_runs = add_accuracy_percent(to_runs_df(load_payload(BASELINE_NPY)))
    fedala_runs = add_accuracy_percent(to_runs_df(load_payload(FEDALA_NPY)))
    scal_runs = add_accuracy_percent(to_runs_df(load_payload(SCALABILITY_NPY)))

    focus_ds = ["MUTAG", "BZR", "COX2", "AIDS"]

    t1 = build_table1_baselines_vs_paper(baseline_runs)
    t1_focus = t1[t1["dataset"].isin(focus_ds)].reset_index(drop=True)

    t2 = build_table2_k10_fedavg_fedala_fedalar(baseline_runs, fedala_runs, datasets=focus_ds)
    t3 = build_table3_scalability(scal_runs, datasets=focus_ds)

    print("\n================ TABLE 1: Baselines vs Paper =================")
    print(t1_focus.to_string(index=False))

    print("\n================ TABLE 2: K=10 FedAvg vs FedALA vs FedALA-R =================")
    print(t2.to_string(index=False))

    print("\n================ TABLE 3: Scalability (client size K) =================")
    print(t3.to_string(index=False))

    save_table(t1_focus, "table1_baselines_vs_paper_focus")
    save_table(t2, "table2_k10_fedavg_fedala_fedalar")
    save_table(t3, "table3_scalability_clients")

    out_npy = OUT_DIR / "graphfl_tables_payload.npy"
    np.save(
        str(out_npy),
        {
            "table1": t1_focus.to_dict(orient="records"),
            "table2": t2.to_dict(orient="records"),
            "table3": t3.to_dict(orient="records"),
        },
        allow_pickle=True,
    )
    print(f"[saved] {out_npy}")


if __name__ == "__main__":
    main()
