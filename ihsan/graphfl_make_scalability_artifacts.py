"""
Graph-FL Overleaf Artifacts Generator
- Tables (LaTeX): baseline vs paper, K=10 comparison, scalability (K sweep)
- Figures: accuracy vs #clients (error bars), improvements vs FedAvg

Inputs (repo-relative default paths):
  ihsan/results_graphfl_table6_baselines2.npy
  ihsan/graphfl_fedala_fedalar_results.npy
  ihsan/results_scalability_graphfl_clients.npy

Outputs:
  ihsan/overleaf_graphfl_artifacts/
    tables/*.tex
    figs/*.pdf and *.png
    analysis_payload.npy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# NumPy pickle compat patch, useful if npy saved elsewhere
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


# Paths
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

BASELINE_NPY = REPO_ROOT / "ihsan" / "results_graphfl_table6_baselines2.npy"
FEDALA_NPY = REPO_ROOT / "ihsan" / "graphfl_fedala_fedalar_results.npy"
SCALABILITY_NPY = REPO_ROOT / "ihsan" / "results_scalability_graphfl_clients.npy"

OUT_DIR = REPO_ROOT / "ihsan" / "overleaf_graphfl_artifacts"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figs"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Paper Table 6 references (Graph-FL test acc %, mean±std)
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
PAPER_AIDS_FEDAVG = (94.2, 0.7)


FOCUS_DATASETS = ["BZR", "COX2", "AIDS"]
METHOD_LABEL = {"fedavg": "FedAvg", "fedala": "FedALA", "fedala_r": "FedALA-R"}



# IO / normalization
def load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    obj = np.load(str(path), allow_pickle=True).item()
    if not isinstance(obj, dict) or "runs" not in obj:
        raise ValueError(f"Unexpected .npy payload format in {path}")
    return obj


def to_runs_df(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload.get("runs", []))


def add_acc_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any of:
      - best_test_accuracy_percent  (already percent)
      - best_test_accuracy          (fraction or percent)
    into: acc_percent
    """
    if df.empty:
        df["acc_percent"] = []
        return df

    if "best_test_accuracy_percent" in df.columns:
        df["acc_percent"] = pd.to_numeric(df["best_test_accuracy_percent"], errors="coerce")
        return df

    if "best_test_accuracy" not in df.columns:
        raise KeyError(
            "No accuracy column found. Expected 'best_test_accuracy' or 'best_test_accuracy_percent'. "
            f"Columns: {list(df.columns)}"
        )

    s = pd.to_numeric(df["best_test_accuracy"], errors="coerce")
    mx = float(s.dropna().max()) if not s.dropna().empty else 0.0
    df["acc_percent"] = s * 100.0 if mx <= 1.2 else s
    return df


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    mean/std/n over groups, ignore NaNs.
    std uses ddof=0 to match your console summaries.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "std", "n"])

    g = (
        df.groupby(group_cols, dropna=False)["acc_percent"]
        .agg(
            mean="mean",
            std=lambda x: float(x.std(ddof=0)) if len(x.dropna()) > 1 else 0.0,
            n="count",
        )
        .reset_index()
    )
    return g


def fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"


def write_tex(path: Path, tex: str) -> None:
    path.write_text(tex, encoding="utf-8")
    print(f"[saved] {path}")


# LaTeX table builders
def latex_table_baselines_vs_paper(baseline_df: pd.DataFrame) -> str:
    s = summarize(baseline_df, ["method", "dataset"])
    s = s[s["dataset"].isin(FOCUS_DATASETS)].copy()

    rows = []
    for _, r in s.iterrows():
        method = str(r["method"])
        dataset = str(r["dataset"])
        ours_m, ours_s = float(r["mean"]), float(r["std"])

        paper_m, paper_s = float("nan"), float("nan")
        if dataset == "AIDS" and method == "fedavg":
            paper_m, paper_s = PAPER_AIDS_FEDAVG
        elif method in PAPER_TABLE6 and dataset in PAPER_TABLE6[method]:
            paper_m, paper_s = PAPER_TABLE6[method][dataset]

        delta = ours_m - paper_m if not pd.isna(paper_m) else float("nan")
        delta_str = f"{delta:+.2f}" if not pd.isna(delta) else "--"

        rows.append((METHOD_LABEL.get(method, method), dataset, fmt_pm(paper_m, paper_s), fmt_pm(ours_m, ours_s), delta_str, int(r["n"])))

    # Sort for readability
    rows.sort(key=lambda x: (x[0], x[1]))

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{5pt}")
    tex.append("\\caption{Baseline comparison against OpenFGL Table~6 (Graph-FL). AIDS paper reference is FedAvg only.}")
    tex.append("\\label{tab:graphfl_baselines_vs_paper}")
    tex.append("\\begin{tabular}{llcccc}")
    tex.append("\\toprule")
    tex.append("\\textbf{Method} & \\textbf{Dataset} & \\textbf{Paper (\\%)} & \\textbf{Ours (\\%)} & \\textbf{$\\Delta$ Mean} & \\textbf{$n$}\\\\")
    tex.append("\\midrule")
    for m, d, paper_cell, ours_cell, delta_cell, n in rows:
        tex.append(f"{m} & {d} & {paper_cell} & {ours_cell} & {delta_cell} & {n}\\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    return "\n".join(tex)


def latex_table_k10_fedavg_fedala_fedalar(baseline_df: pd.DataFrame, fedala_df: pd.DataFrame) -> str:
    df = pd.concat(
        [
            baseline_df[baseline_df["method"].isin(["fedavg"])],
            fedala_df[fedala_df["method"].isin(["fedala", "fedala_r"])],
        ],
        ignore_index=True,
    )
    df = df[df["dataset"].isin(FOCUS_DATASETS)].copy()

    s = summarize(df, ["method", "dataset"])
    # Build a pivot for cells
    cell = {(r["method"], r["dataset"]): fmt_pm(float(r["mean"]), float(r["std"])) for _, r in s.iterrows()}

    methods = ["fedavg", "fedala", "fedala_r"]
    datasets = FOCUS_DATASETS

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{5pt}")
    tex.append("\\caption{FedAvg vs FedALA vs FedALA-R on Graph-FL with $K=10$ clients (mean$\\pm$std, \\% accuracy).}")
    tex.append("\\label{tab:graphfl_k10_fedavg_fedala_fedalar}")
    tex.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    tex.append("\\toprule")
    tex.append("\\textbf{Method} & " + " & ".join([f"\\textbf{{{d}}}" for d in datasets]) + "\\\\")
    tex.append("\\midrule")
    for m in methods:
        row = [f"\\textbf{{{METHOD_LABEL[m]}}}"]
        for d in datasets:
            row.append(cell.get((m, d), "--"))
        tex.append(" & ".join(row) + "\\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    return "\n".join(tex)


def latex_table_scalability(scal_df: pd.DataFrame) -> str:
    df = scal_df.copy()
    df = df[df["dataset"].isin(FOCUS_DATASETS)]
    df = df[df["method"].isin(["fedavg", "fedala", "fedala_r"])]

    s = summarize(df, ["dataset", "method", "K"])
    # cells keyed by (dataset, method, K)
    cell = {(r["dataset"], r["method"], int(r["K"])): fmt_pm(float(r["mean"]), float(r["std"])) for _, r in s.iterrows()}
    Ks = sorted(int(k) for k in s["K"].dropna().unique())

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\scriptsize")
    tex.append("\\setlength{\\tabcolsep}{4pt}")
    tex.append("\\caption{Scalability on Graph-FL: accuracy (mean$\\pm$std, \\%) across different client counts $K$.}")
    tex.append("\\label{tab:graphfl_scalability}")
    tex.append("\\begin{tabular}{ll" + "c" * len(Ks) + "}")
    tex.append("\\toprule")
    tex.append("\\textbf{Dataset} & \\textbf{Method} & " + " & ".join([f"\\textbf{{K={k}}}" for k in Ks]) + "\\\\")
    tex.append("\\midrule")

    for di, dataset in enumerate(FOCUS_DATASETS):
        for mi, method in enumerate(["fedavg", "fedala", "fedala_r"]):
            left = dataset if mi == 0 else ""
            row = [left, METHOD_LABEL[method]]
            for k in Ks:
                row.append(cell.get((dataset, method, k), "--"))
            tex.append(" & ".join(row) + "\\\\")
        if di != len(FOCUS_DATASETS) - 1:
            tex.append("\\midrule")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    return "\n".join(tex)


# Plot builders
def plot_accuracy_vs_clients_scalability(
    scal_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    df = scal_df.copy()
    df = df[df["dataset"].isin(FOCUS_DATASETS)]
    df = df[df["method"].isin(METHOD_LABEL)]

    s = summarize(df, ["dataset", "method", "K"])

    # Ensure K is integer and sorted; x-axis must be multiples of 5
    Ks = sorted(int(k) for k in s["K"].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)

    for j, dataset in enumerate(FOCUS_DATASETS):
        ax = axes[j]

        for method in METHOD_LABEL:
            sub = s[(s["dataset"] == dataset) & (s["method"] == method)].copy()
            if sub.empty:
                continue
            sub["K"] = sub["K"].astype(int)
            sub = sub.sort_values("K")

            x = sub["K"].to_list()
            y = sub["mean"].astype(float).to_list()
            yerr = sub["std"].astype(float).to_list()

            ax.errorbar(
                x, y, yerr=yerr,
                marker="o",
                capsize=3,
                label=METHOD_LABEL.get(method, method),
            )

        ax.set_title(dataset)
        ax.set_xlabel("Clients")
        ax.set_xticks([k for k in Ks if k % 5 == 0])  # multiples of 5
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if j == 0:
            ax.set_ylabel("Accuracy (%)")
        else:
            ax.set_ylabel("")

        # Put legend only in ONE subplot (rightmost), like your example
        if j == 2:
            ax.legend(frameon=False, loc="best")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")



def plot_improvement_vs_fedavg(scal_df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    """
    Two improvements:
      - FedALA - FedAvg (absolute points)
      - FedALA-R - FedAvg
    shown vs K for each dataset.
    """
    df = scal_df.copy()
    df = df[df["dataset"].isin(FOCUS_DATASETS)]
    df = df[df["method"].isin(["fedavg", "fedala", "fedala_r"])]

    s = summarize(df, ["dataset", "method", "K"])

    # create a pivot for means
    piv = s.pivot_table(index=["dataset", "K"], columns="method", values="mean", aggfunc="first").reset_index()
    piv["imp_fedala"] = piv["fedala"] - piv["fedavg"]
    piv["imp_fedala_r"] = piv["fedala_r"] - piv["fedavg"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, FOCUS_DATASETS):
        sub = piv[piv["dataset"] == dataset].sort_values("K")
        if sub.empty:
            ax.set_title(dataset)
            ax.axis("off")
            continue

        x = sub["K"].astype(int).to_list()
        ax.plot(x, sub["imp_fedala"].astype(float).to_list(), marker="o", label="FedALA − FedAvg")
        ax.plot(x, sub["imp_fedala_r"].astype(float).to_list(), marker="o", label="FedALA-R − FedAvg")
        ax.axhline(0.0, linewidth=1.0)

        ax.set_title(dataset)
        ax.set_xlabel("Clients (K)")
        ax.set_ylabel("Improvement (points)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


def plot_k10_relative_improvement_bar(
    baseline_df: pd.DataFrame,
    fedala_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """
    For K=10, show RELATIVE improvement over FedAvg:
      (method_mean - fedavg_mean) / fedavg_mean * 100
    for FedALA and FedALA-R, per dataset.
    """
    df = pd.concat(
        [
            baseline_df[baseline_df["method"].isin(["fedavg"])],
            fedala_df[fedala_df["method"].isin(["fedala", "fedala_r"])],
        ],
        ignore_index=True,
    )
    df = df[df["dataset"].isin(FOCUS_DATASETS)]

    s = summarize(df, ["dataset", "method"])

    piv = s.pivot_table(index="dataset", columns="method", values="mean", aggfunc="first")
    piv = piv.reindex(FOCUS_DATASETS)

    rel_ala = (piv["fedala"] - piv["fedavg"]) / piv["fedavg"] * 100.0
    rel_alr = (piv["fedala_r"] - piv["fedavg"]) / piv["fedavg"] * 100.0

    x = np.arange(len(FOCUS_DATASETS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    ax.bar(x - width / 2, rel_ala.values, width, label="FedALA vs FedAvg (relative %)")
    ax.bar(x + width / 2, rel_alr.values, width, label="FedALA-R vs FedAvg (relative %)")

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(FOCUS_DATASETS)
    ax.set_ylabel("Relative improvement (%)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


# Main
def main() -> None:
    baseline_runs = add_acc_percent(to_runs_df(load_payload(BASELINE_NPY)))
    fedala_runs = add_acc_percent(to_runs_df(load_payload(FEDALA_NPY)))
    scal_runs = add_acc_percent(to_runs_df(load_payload(SCALABILITY_NPY)))

    # --- Tables ---
    tex_t1 = latex_table_baselines_vs_paper(baseline_runs)
    tex_t2 = latex_table_k10_fedavg_fedala_fedalar(baseline_runs, fedala_runs)
    tex_t3 = latex_table_scalability(scal_runs)

    write_tex(TABLE_DIR / "table1_baselines_vs_paper_focus.tex", tex_t1)
    write_tex(TABLE_DIR / "table2_k10_fedavg_fedala_fedalar.tex", tex_t2)
    write_tex(TABLE_DIR / "table3_scalability.tex", tex_t3)

    # --- Figures (Scalability only: accuracy vs clients; 1x3; no MUTAG) ---
    plot_accuracy_vs_clients_scalability(
        scal_runs,
        out_pdf=FIG_DIR / "fig_graphfl_scalability_accuracy_vs_clients.pdf",
        out_png=FIG_DIR / "fig_graphfl_scalability_accuracy_vs_clients.png",
    )

    # plot_improvement_vs_fedavg(...)
    # plot_k10_relative_improvement_bar(...)

    # Optional consolidated payload for future scripts
    out_payload = OUT_DIR / "analysis_payload.npy"
    np.save(
        str(out_payload),
        {
            "meta": {
                "focus_datasets": FOCUS_DATASETS,
                "inputs": {
                    "baseline_npy": str(BASELINE_NPY),
                    "fedala_npy": str(FEDALA_NPY),
                    "scalability_npy": str(SCALABILITY_NPY),
                },
            },
        },
        allow_pickle=True,
    )
    print(f"[saved] {out_payload}")

if __name__ == "__main__":
    main()
