"""
ihsan/runtime_artifacts_generation_graphfl.py

Generates Graph-FL runtime + overhead artifacts from:
  results_runtime_graphfl_clients.npy

Outputs:
  ihsan/overleaf_graphfl_runtime_artifacts/
    figs/graphfl_runtime_overhead.png
    figs/graphfl_runtime_overhead.pdf
    tables/table_runtime_overhead_graphfl.tex
    tables/table_runtime_overhead_graphfl.csv

Notes:
- Expects per-run keys (as in your payload):
    wall_time_sec_per_round
    overhead_sec_per_round_vs_fedavg
    best_test_accuracy_percent
- If overhead is missing, it will be derived by matching FedAvg runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

IN_NPY = REPO_ROOT / "results_runtime_graphfl_clients.npy"

OUT_DIR = REPO_ROOT / "ihsan" / "overleaf_graphfl_runtime_artifacts"
FIG_DIR = OUT_DIR / "figs"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
DATASETS = ["BZR", "COX2", "AIDS"]
METHODS_ALL = ["fedavg", "fedala", "fedala_r"]
METHODS_OVERHEAD = ["fedala", "fedala_r"]
METHOD_LABEL = {"fedavg": "FedAvg", "fedala": "FedALA", "fedala_r": "FedALA-R"}

CLIENT_SIZES = [5, 10, 15, 20]


# -----------------------------
# IO
# -----------------------------
def load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    obj = np.load(str(path), allow_pickle=True).item()
    if not isinstance(obj, dict) or "runs" not in obj:
        raise ValueError(f"Unexpected payload format in {path} (expected dict with 'runs').")
    return obj


def to_df(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("runs", []))
    if df.empty:
        raise ValueError("No runs found in payload.")
    return df


# -----------------------------
# Normalization / derivations
# -----------------------------
def infer_time_per_round_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer wall_time_sec_per_round (your payload), otherwise derive total/rounds.
    """
    df = df.copy()

    if "wall_time_sec_per_round" in df.columns:
        df["time_per_round_sec"] = pd.to_numeric(df["wall_time_sec_per_round"], errors="coerce")
        return df

    # fallbacks
    candidates = [
        "time_per_round_sec",
        "time_per_round",
        "avg_round_time_sec",
        "avg_time_per_round_sec",
        "mean_round_time_sec",
    ]
    for k in candidates:
        if k in df.columns:
            df["time_per_round_sec"] = pd.to_numeric(df[k], errors="coerce")
            return df

    if ("wall_time_sec_total" in df.columns) and ("num_rounds" in df.columns):
        tot = pd.to_numeric(df["wall_time_sec_total"], errors="coerce")
        nr = pd.to_numeric(df["num_rounds"], errors="coerce")
        df["time_per_round_sec"] = tot / nr.replace({0: np.nan})
        return df

    raise KeyError(
        "Could not find per-round runtime in runs. Expected 'wall_time_sec_per_round' "
        "or derivation from wall_time_sec_total/num_rounds."
    )


def infer_overhead_vs_fedavg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer overhead_sec_per_round_vs_fedavg (your payload).
    If missing, compute overhead by matching FedAvg on (dataset, K, seed, instance_index, repeat, is_warmup).
    """
    df = df.copy()

    if "overhead_sec_per_round_vs_fedavg" in df.columns:
        df["overhead_sec_per_round_vs_fedavg"] = pd.to_numeric(
            df["overhead_sec_per_round_vs_fedavg"], errors="coerce"
        )
        return df

    # Derive overhead by matching to FedAvg (best-effort)
    join_cols = [c for c in ["dataset", "K", "seed", "instance_index", "repeat", "is_warmup"] if c in df.columns]
    if not join_cols:
        raise KeyError(
            "Cannot derive overhead (no overhead key present and no join columns available)."
        )

    fedavg = df[df["method"] == "fedavg"][join_cols + ["time_per_round_sec"]].copy()
    fedavg = fedavg.rename(columns={"time_per_round_sec": "fedavg_time_per_round_sec"})

    merged = df.merge(fedavg, on=join_cols, how="left")
    merged["overhead_sec_per_round_vs_fedavg"] = (
        merged["time_per_round_sec"] - merged["fedavg_time_per_round_sec"]
    )
    return merged


def summarize_mean_std(df: pd.DataFrame, value_col: str, group_cols: List[str]) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(
            mean="mean",
            std=lambda x: float(x.std(ddof=0)) if x.dropna().shape[0] > 1 else 0.0,
            n="count",
        )
        .reset_index()
    )
    return g


# -----------------------------
# Table builder
# -----------------------------
def build_overhead_table(
    time_summary: pd.DataFrame,
    overhead_summary: pd.DataFrame,
    ks: List[int],
) -> pd.DataFrame:
    """
    Produces a table like:
      Dataset | FedALA Overhead | FedALA-R Overhead
    where each cell is: "<sec>s (<pct>%)"
    averaged over K in ks.
    """
    # FedAvg baseline time (mean) per dataset,K
    fedavg_t = time_summary[time_summary["method"] == "fedavg"][["dataset", "K", "mean"]].copy()
    fedavg_t = fedavg_t.rename(columns={"mean": "fedavg_time_mean"})

    # overhead means per dataset,K,method
    oh = overhead_summary[overhead_summary["method"].isin(METHODS_OVERHEAD)][["dataset", "method", "K", "mean"]].copy()
    oh = oh.rename(columns={"mean": "overhead_mean"})

    # merge to compute percentage vs FedAvg at each K
    m = oh.merge(fedavg_t, on=["dataset", "K"], how="left")
    m["overhead_pct_vs_fedavg"] = (m["overhead_mean"] / m["fedavg_time_mean"]) * 100.0

    # restrict to desired Ks
    m = m[m["K"].isin(ks)].copy()

    # average over Ks (simple mean across the K points)
    agg = (
        m.groupby(["dataset", "method"], dropna=False)
        .agg(
            overhead_sec=("overhead_mean", "mean"),
            overhead_pct=("overhead_pct_vs_fedavg", "mean"),
        )
        .reset_index()
    )

    # pivot into wide
    piv = agg.pivot(index="dataset", columns="method", values=["overhead_sec", "overhead_pct"])
    piv = piv.reindex(DATASETS)

    def cell(ds: str, method: str) -> str:
        sec = piv.loc[ds, ("overhead_sec", method)]
        pct = piv.loc[ds, ("overhead_pct", method)]
        if pd.isna(sec) or pd.isna(pct):
            return "--"
        return f"{sec:.3f}s ({pct:.1f}\\%)"

    out = pd.DataFrame(
        {
            "Dataset": DATASETS,
            "FedALA Overhead": [cell(d, "fedala") for d in DATASETS],
            "FedALA-R Overhead": [cell(d, "fedala_r") for d in DATASETS],
        }
    )
    return out


def write_latex_table_overhead(df_table: pd.DataFrame, out_tex: Path) -> None:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\caption{Average per-round runtime overhead vs. FedAvg on Graph-FL, averaged over $K \\in \\{5,10,15,20\\}$.}")
    lines.append("\\label{tab:graphfl_runtime_overhead}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{FedALA Overhead} & \\textbf{FedALA-R Overhead}\\\\")
    lines.append("\\midrule")
    for _, r in df_table.iterrows():
        lines.append(f"{r['Dataset']} & {r['FedALA Overhead']} & {r['FedALA-R Overhead']}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Plot builder (3 rows x 2 cols)
# -----------------------------
def plot_runtime_and_overhead(
    time_summary: pd.DataFrame,
    overhead_summary: pd.DataFrame,
    ks: List[int],
    out_png: Path,
    out_pdf: Path,
) -> None:
    # enforce ordering
    ks = [k for k in ks if k in CLIENT_SIZES]

    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=2,
        figsize=(10.5, 9.0),
        constrained_layout=True,
    )

    for row_i, dataset in enumerate(DATASETS):
        ax_time = axes[row_i, 0]
        ax_oh = axes[row_i, 1]

        # ---- runtime plot (FedAvg, FedALA, FedALA-R)
        for method in METHODS_ALL:
            sub = time_summary[
                (time_summary["dataset"] == dataset)
                & (time_summary["method"] == method)
                & (time_summary["K"].isin(ks))
            ].sort_values("K")

            if sub.empty:
                continue

            x = sub["K"].astype(int).tolist()
            y = sub["mean"].astype(float).tolist()
            yerr = sub["std"].astype(float).tolist()

            ax_time.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=METHOD_LABEL[method])

        ax_time.set_title(f"(a) {dataset}: Runtime vs Client Size")
        ax_time.set_xlabel("Client size")
        ax_time.set_ylabel("Time per run (sec)")
        ax_time.set_xticks(ks)
        ax_time.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # ---- overhead plot (FedALA, FedALA-R), vs FedAvg
        for method in METHODS_OVERHEAD:
            sub = overhead_summary[
                (overhead_summary["dataset"] == dataset)
                & (overhead_summary["method"] == method)
                & (overhead_summary["K"].isin(ks))
            ].sort_values("K")

            if sub.empty:
                continue

            x = sub["K"].astype(int).tolist()
            y = sub["mean"].astype(float).tolist()
            yerr = sub["std"].astype(float).tolist()

            label = f"{METHOD_LABEL[method]} − {METHOD_LABEL['fedavg']}"
            ax_oh.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)

        ax_oh.axhline(0.0, linewidth=1.0, linestyle="--")
        ax_oh.set_title(f"(b) {dataset}: Overhead vs FedAvg")
        ax_oh.set_xlabel("Client size")
        ax_oh.set_ylabel("Extra time (sec)")
        ax_oh.set_xticks(ks)
        ax_oh.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # Put legends only on the first row (matches your example intent)
        if row_i == 0:
            ax_time.legend(loc="upper left", frameon=True, fontsize=9)
            ax_oh.legend(loc="upper left", frameon=True, fontsize=9)

    fig.savefig(out_png, dpi=250)
    fig.savefig(out_pdf)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    payload = load_payload(IN_NPY)
    runs = to_df(payload)

    # basic filtering
    runs = runs[runs["dataset"].isin(DATASETS)].copy()
    runs = runs[runs["method"].isin(METHODS_ALL)].copy()
    runs["K"] = pd.to_numeric(runs["K"], errors="coerce").astype("Int64")

    # infer runtime + overhead
    runs = infer_time_per_round_sec(runs)
    runs = infer_overhead_vs_fedavg(runs)

    # Keep the standard scalability grid only
    runs = runs[runs["K"].isin(CLIENT_SIZES)].copy()

    # summarize (mean±std) by dataset/method/K
    time_sum = summarize_mean_std(runs, "time_per_round_sec", ["dataset", "method", "K"])
    oh_sum = summarize_mean_std(runs, "overhead_sec_per_round_vs_fedavg", ["dataset", "method", "K"])

    # ---- Figure (3x2)
    out_png = FIG_DIR / "graphfl_runtime_overhead.png"
    out_pdf = FIG_DIR / "graphfl_runtime_overhead.pdf"
    plot_runtime_and_overhead(time_sum, oh_sum, CLIENT_SIZES, out_png, out_pdf)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")

    # ---- Table (avg overhead over K)
    tbl = build_overhead_table(time_sum, oh_sum, CLIENT_SIZES)
    out_csv = TAB_DIR / "table_runtime_overhead_graphfl.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    out_tex = TAB_DIR / "table_runtime_overhead_graphfl.tex"
    write_latex_table_overhead(tbl, out_tex)
    print(f"[saved] {out_tex}")

    print("\nOverleaf usage:")
    print("  - Copy figs/*.png (or *.pdf) into Overleaf and include with \\includegraphics")
    print("  - Copy tables/*.tex and use \\input{tables/table_runtime_overhead_graphfl.tex}")


if __name__ == "__main__":
    main()
