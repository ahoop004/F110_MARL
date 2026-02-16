#!/usr/bin/env python3
"""Analyze spawn-position correlations from a W&B run.

Generates:
1) Spawn vs shaping reward scatter + binned trend
2) Spawn vs success-rate-by-bin plot
3) Lagged correlation plot: spawn vs shaping reward
4) Lagged correlation plot: spawn vs success

Usage:
  python3 tools/analyze_spawn_correlations.py \
      --entity <entity> --project <project> --run-id <run_id_or_name>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr, spearmanr
import wandb
from wandb.errors import UsageError


DEFAULT_SPAWN_CANDIDATES = (
    "train/spawn_y",
    "spawn_y",
)
DEFAULT_SUCCESS_CANDIDATES = (
    "train/episode_success",
    "train/success",
    "train/success_rate",
)
DEFAULT_SHAPING_CANDIDATES = (
    "train/reward_shaping",
    "train/reward_gaplock_pressure_shaping",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spawn-vs-reward/success correlations from W&B.")
    parser.add_argument("--entity", type=str, default="", help="W&B entity (team/user).")
    parser.add_argument("--project", type=str, default="", help="W&B project.")
    parser.add_argument("--run-id", type=str, default="", help="Run id or run name.")
    parser.add_argument("--run-path", type=str, default="", help="Full run path entity/project/run.")
    parser.add_argument(
        "--history-csv",
        type=str,
        default="",
        help="Use local W&B history CSV instead of API (skips login).",
    )
    parser.add_argument("--step-col", type=str, default="train/episode", help="Episode step column.")
    parser.add_argument("--spawn-col", type=str, default="", help="Spawn column (default: auto-detect).")
    parser.add_argument("--success-col", type=str, default="", help="Success column (default: auto-detect).")
    parser.add_argument("--shaping-col", type=str, default="", help="Shaping reward column (default: auto-detect).")
    parser.add_argument("--bins", type=int, default=20, help="Number of spawn bins.")
    parser.add_argument("--max-lag", type=int, default=50, help="Max lag for lagged-correlation plots.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots/analysis",
        help="Output directory (run-specific subdir will be created).",
    )
    return parser.parse_args()


def _resolve_run_path(args: argparse.Namespace) -> str:
    if args.run_path.strip():
        return args.run_path.strip()
    if not args.entity.strip() or not args.project.strip() or not args.run_id.strip():
        raise ValueError("Provide either --run-path or (--entity, --project, --run-id).")
    return f"{args.entity.strip()}/{args.project.strip()}/{args.run_id.strip()}"


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return pd.to_numeric(series, errors="coerce")


def _pick_best_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    best_col = None
    best_count = -1
    for col in candidates:
        if col not in df.columns:
            continue
        count = int(_to_numeric(df[col]).notna().sum())
        if count > best_count:
            best_col = col
            best_count = count
    return best_col


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y)[0])


def _safe_pointbiserial(binary: np.ndarray, continuous: np.ndarray) -> float:
    if binary.size < 3 or continuous.size < 3:
        return float("nan")
    vals = np.unique(binary)
    if vals.size < 2:
        return float("nan")
    if np.allclose(continuous, continuous[0]):
        return float("nan")
    return float(pointbiserialr(binary, continuous)[0])


def _lagged_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: int,
    binary_y: bool = False,
) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xa = x[-lag:]
            ya = y[: lag or None]
        elif lag > 0:
            xa = x[:-lag]
            ya = y[lag:]
        else:
            xa = x
            ya = y

        if binary_y:
            corr = _safe_pointbiserial(ya.astype(int), xa.astype(float))
        else:
            corr = _safe_corr(xa.astype(float), ya.astype(float))
        rows.append({"lag": lag, "corr": corr, "n": int(xa.size)})
    return pd.DataFrame(rows)


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if math.isclose(lo, hi):
        hi = lo + 1e-6
    return np.linspace(lo, hi, bins + 1)


def _plot_spawn_vs_shaping(df: pd.DataFrame, spawn_col: str, shaping_col: str, out: Path, bins: int) -> None:
    clean = df[[spawn_col, shaping_col]].dropna()
    if clean.empty:
        return

    x = clean[spawn_col].to_numpy(dtype=float)
    y = clean[shaping_col].to_numpy(dtype=float)
    edges = _bin_edges(x, bins)
    binned = pd.DataFrame({"x": x, "y": y})
    binned["bin"] = pd.cut(binned["x"], edges, include_lowest=True)
    trend = binned.groupby("bin", observed=True).agg(x_mid=("x", "mean"), y_mean=("y", "mean"))

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=8, alpha=0.2, label="episodes")
    if not trend.empty:
        plt.plot(trend["x_mid"], trend["y_mean"], linewidth=2.5, label="binned mean")
    plt.xlabel(spawn_col)
    plt.ylabel(shaping_col)
    plt.title("Spawn vs Shaping Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_spawn_vs_success(df: pd.DataFrame, spawn_col: str, success_col: str, out: Path, bins: int) -> None:
    clean = df[[spawn_col, success_col]].dropna()
    if clean.empty:
        return

    x = clean[spawn_col].to_numpy(dtype=float)
    s = clean[success_col].to_numpy(dtype=float)
    s = (s > 0.5).astype(int)

    edges = _bin_edges(x, bins)
    binned = pd.DataFrame({"x": x, "s": s})
    binned["bin"] = pd.cut(binned["x"], edges, include_lowest=True)
    agg = binned.groupby("bin", observed=True).agg(x_mid=("x", "mean"), p=("s", "mean"), n=("s", "size"))
    if agg.empty:
        return
    se = np.sqrt(np.clip(agg["p"] * (1.0 - agg["p"]) / np.maximum(agg["n"], 1), 0.0, None))

    plt.figure(figsize=(9, 6))
    plt.errorbar(agg["x_mid"], agg["p"], yerr=se, fmt="-o", linewidth=2, markersize=4, capsize=2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel(spawn_col)
    plt.ylabel("success rate")
    plt.title("Spawn vs Success Rate (Binned)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_lag(df: pd.DataFrame, out: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(df["lag"], df["corr"], linewidth=2)
    plt.axhline(0.0, linewidth=1, alpha=0.5)
    plt.axvline(0, linewidth=1, alpha=0.5)
    plt.xlabel("lag (episodes)")
    plt.ylabel("correlation")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _build_episode_frame(df: pd.DataFrame, step_col: str) -> pd.DataFrame:
    working = df.copy()
    if step_col not in working.columns:
        if "_step" in working.columns:
            step_col = "_step"
        else:
            working["_episode_index"] = np.arange(len(working), dtype=int)
            step_col = "_episode_index"
    working = working.sort_values(step_col)
    # Keep last row per episode, which usually contains episode-level metrics.
    return working.groupby(step_col, as_index=False).last()


def main() -> None:
    args = _parse_args()
    run_path = ""
    run = None
    if args.history_csv.strip():
        csv_path = Path(args.history_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"History CSV not found: {csv_path}")
        df_raw = pd.read_csv(csv_path)
        run_slug = csv_path.stem
    else:
        run_path = _resolve_run_path(args)
        try:
            api = wandb.Api()
            run = api.run(run_path)
        except UsageError as exc:
            raise RuntimeError(
                "W&B API key is not configured. "
                "Run `wandb login` first, or pass --history-csv <exported_history.csv>."
            ) from exc
        rows = list(run.scan_history())
        if not rows:
            raise RuntimeError(f"No history rows found for run: {run_path}")
        df_raw = pd.DataFrame(rows)
        run_slug = run.id or run.name or "run"

    df_ep = _build_episode_frame(df_raw, args.step_col)

    spawn_col = args.spawn_col.strip() or _pick_best_column(df_ep, DEFAULT_SPAWN_CANDIDATES)
    success_col = args.success_col.strip() or _pick_best_column(df_ep, DEFAULT_SUCCESS_CANDIDATES)

    shaping_col = args.shaping_col.strip()
    if not shaping_col:
        shaping_col = _pick_best_column(df_ep, DEFAULT_SHAPING_CANDIDATES) or ""
        if not shaping_col:
            c_col = "train/reward_gaplock_pressure_centerline"
            w_col = "train/reward_gaplock_pressure_wall"
            if c_col in df_ep.columns and w_col in df_ep.columns:
                df_ep["train/reward_shaping"] = _to_numeric(df_ep[c_col]).fillna(0.0) + _to_numeric(
                    df_ep[w_col]
                ).fillna(0.0)
                shaping_col = "train/reward_shaping"

    if not spawn_col:
        raise RuntimeError(
            "Could not find spawn column. Expected one of: "
            + ", ".join(DEFAULT_SPAWN_CANDIDATES)
        )
    if not success_col:
        raise RuntimeError(
            "Could not find success column. Expected one of: "
            + ", ".join(DEFAULT_SUCCESS_CANDIDATES)
        )

    df_ep[spawn_col] = _to_numeric(df_ep[spawn_col])
    df_ep[success_col] = _to_numeric(df_ep[success_col])
    if shaping_col:
        df_ep[shaping_col] = _to_numeric(df_ep[shaping_col])

    outdir = Path(args.outdir).expanduser().resolve() / f"{run_slug}_spawn_corr"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_path": run_path,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "num_rows_raw": int(len(df_raw)),
        "num_rows_episode": int(len(df_ep)),
        "spawn_col": spawn_col,
        "success_col": success_col,
        "shaping_col": shaping_col if shaping_col else None,
    }

    # Spawn vs shaping
    if shaping_col:
        clean = df_ep[[spawn_col, shaping_col]].dropna()
        if len(clean) >= 3:
            rho, p = spearmanr(clean[spawn_col].to_numpy(dtype=float), clean[shaping_col].to_numpy(dtype=float))
            summary["spawn_vs_shaping_spearman_rho"] = float(rho)
            summary["spawn_vs_shaping_spearman_p"] = float(p)
            summary["spawn_vs_shaping_n"] = int(len(clean))

            _plot_spawn_vs_shaping(
                df_ep,
                spawn_col=spawn_col,
                shaping_col=shaping_col,
                out=outdir / "spawn_vs_shaping_scatter_binned.png",
                bins=max(4, int(args.bins)),
            )

            lag_reward = _lagged_corr(
                clean[spawn_col].to_numpy(dtype=float),
                clean[shaping_col].to_numpy(dtype=float),
                max_lag=max(1, int(args.max_lag)),
                binary_y=False,
            )
            _plot_lag(
                lag_reward,
                out=outdir / "lagged_corr_spawn_vs_shaping.png",
                title="Lagged Corr: spawn vs shaping reward",
            )
            lag_reward.to_csv(outdir / "lagged_corr_spawn_vs_shaping.csv", index=False)
        else:
            summary["spawn_vs_shaping_note"] = "Insufficient paired samples for shaping analysis."
    else:
        summary["spawn_vs_shaping_note"] = "Shaping reward column not found; shaping analysis skipped."

    # Spawn vs success
    clean_success = df_ep[[spawn_col, success_col]].dropna()
    if len(clean_success) >= 3:
        s = (clean_success[success_col].to_numpy(dtype=float) > 0.5).astype(int)
        x = clean_success[spawn_col].to_numpy(dtype=float)
        r = _safe_pointbiserial(s, x)
        summary["spawn_vs_success_pointbiserial_r"] = float(r) if not np.isnan(r) else None
        summary["spawn_vs_success_n"] = int(len(clean_success))
        summary["success_positive_rate"] = float(np.mean(s))

        _plot_spawn_vs_success(
            df_ep,
            spawn_col=spawn_col,
            success_col=success_col,
            out=outdir / "spawn_vs_success_binned.png",
            bins=max(4, int(args.bins)),
        )

        lag_success = _lagged_corr(
            x,
            s.astype(float),
            max_lag=max(1, int(args.max_lag)),
            binary_y=True,
        )
        _plot_lag(
            lag_success,
            out=outdir / "lagged_corr_spawn_vs_success.png",
            title="Lagged Corr: spawn vs success",
        )
        lag_success.to_csv(outdir / "lagged_corr_spawn_vs_success.csv", index=False)
    else:
        summary["spawn_vs_success_note"] = "Insufficient paired samples for success analysis."

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[done] wrote outputs to: {outdir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
