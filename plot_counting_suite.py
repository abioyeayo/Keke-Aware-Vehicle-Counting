from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd


DOMINANT_CLASSES: Tuple[str, ...] = ("bus", "car", "keke", "truck")
DIRECTIONS: Tuple[str, ...] = ("downward", "upward")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def load_suite_index(suite_dir: Path) -> pd.DataFrame:
    index_path = suite_dir / "experiment_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)

    records: List[dict] = []
    for summary_path in suite_dir.glob("**/summary.json"):
        try:
            records.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not records:
        raise FileNotFoundError(f"Could not find experiment_index.csv or any summary.json files under {suite_dir}")
    return pd.DataFrame(records)



def load_segments(summary_row: pd.Series) -> pd.DataFrame:
    outputs = summary_row.get("outputs")
    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except json.JSONDecodeError:
            try:
                outputs = ast.literal_eval(outputs)
            except (ValueError, SyntaxError):
                outputs = None

    candidate = None
    if isinstance(outputs, dict):
        candidate = outputs.get("segments_csv")
    if not candidate:
        # fallback to expected relative path if suite was moved
        summary_json = None
        if isinstance(outputs, dict):
            summary_json = outputs.get("summary_json")
        if summary_json:
            candidate = str(Path(summary_json).with_name("region_segments.csv"))

    if not candidate:
        raise FileNotFoundError(f"No segments_csv path found for experiment {summary_row.get('experiment_key')}")

    path = Path(candidate)
    if not path.exists():
        # try resolving relative to suite root movement
        alt = Path(summary_row["suite_dir"]) / summary_row["video_key"] / summary_row["experiment_key"] / "region_segments.csv"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Segments CSV not found: {candidate}")
    return pd.read_csv(path)



def load_baseline(path: str, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if path not in cache:
        cache[path] = pd.read_csv(path)
    return cache[path]


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def add_direction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "direction" in out.columns:
        return out
    direction = np.where(
        out["enter_cy"] > out["exit_cy"],
        "upward",
        np.where(out["enter_cy"] < out["exit_cy"], "downward", "stationary"),
    )
    out["direction"] = direction
    return out



def filter_segments(
    segments: pd.DataFrame,
    min_duration_s: float,
    max_duration_s: float,
    min_avg_conf: float,
) -> pd.DataFrame:
    df = add_direction(segments)
    mask = (
        (df["duration_s"] >= min_duration_s)
        & (df["duration_s"] <= max_duration_s)
        & (df["avg_conf_in_region"] >= min_avg_conf)
        & (df["direction"].isin(DIRECTIONS))
    )
    return df.loc[mask].copy()



def greedy_time_match(pred_times: Sequence[float], gt_times: Sequence[float], tol_s: float) -> Tuple[int, int, int, float]:
    pred_sorted = sorted(float(x) for x in pred_times)
    gt_sorted = sorted(float(x) for x in gt_times)
    used = [False] * len(gt_sorted)
    tp = 0
    abs_dt: List[float] = []
    j = 0
    for t in pred_sorted:
        while j < len(gt_sorted) and gt_sorted[j] < t - tol_s:
            j += 1
        candidates = []
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(gt_sorted) and not used[k] and abs(gt_sorted[k] - t) <= tol_s:
                candidates.append((abs(gt_sorted[k] - t), k))
        if candidates:
            dt, k = min(candidates)
            used[k] = True
            tp += 1
            abs_dt.append(float(dt))
    fp = len(pred_sorted) - tp
    fn = len(gt_sorted) - tp
    mean_abs_dt = float(np.mean(abs_dt)) if abs_dt else np.nan
    return tp, fp, fn, mean_abs_dt



def pct_error(pred: int, gt: int) -> float:
    if gt == 0:
        return np.nan
    return 100.0 * (pred - gt) / gt



def abs_pct_error(pred: int, gt: int) -> float:
    if gt == 0:
        return np.nan
    return 100.0 * abs(pred - gt) / gt



def count_accuracy(pred: int, gt: int) -> float:
    if gt == 0:
        return np.nan
    return max(0.0, 1.0 - abs(pred - gt) / gt)



def build_plot_label(row: pd.Series) -> str:
    tracker = row["tracker"] if str(row["tracker"]).endswith("track") else str(row["tracker"]).replace(".yaml", "")
    region = "ROI" if row["region_mode"] == "roi" else "Full frame"
    # region = "ROI" if row["region_mode"] == "roi" else "Full"
    return f"{row['architecture']} {row['variant']} | {tracker} | {region}"


# -----------------------------------------------------------------------------
# Metric extraction
# -----------------------------------------------------------------------------

def compute_metrics(
    suite_index: pd.DataFrame,
    min_duration_s: float,
    max_duration_s: float,
    min_avg_conf: float,
    match_tolerance_s: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_cache: Dict[str, pd.DataFrame] = {}
    count_rows: List[dict] = []
    event_rows: List[dict] = []
    summary_rows: List[dict] = []

    for _, row in suite_index.iterrows():
        segments = load_segments(row)
        filtered = filter_segments(segments, min_duration_s=min_duration_s, max_duration_s=max_duration_s, min_avg_conf=min_avg_conf)
        baseline = load_baseline(row["baseline_path"], baseline_cache)
        plot_label = build_plot_label(row)

        exp_count_rows: List[dict] = []
        exp_event_rows: List[dict] = []

        for direction in DIRECTIONS:
            pred_d = filtered[filtered["direction"] == direction]
            gt_d = baseline[baseline["direction"] == direction]

            for cls_name in DOMINANT_CLASSES:
                pred_c = pred_d[pred_d["class"] == cls_name]
                gt_c = gt_d[gt_d["class"] == cls_name]
                pred_count = int(len(pred_c))
                gt_count = int(len(gt_c))
                abs_error = int(abs(pred_count - gt_count))

                count_row = {
                    "experiment_key": row["experiment_key"],
                    "plot_label": plot_label,
                    "video_key": row["video_key"],
                    "scene": row.get("scene"),
                    "architecture": row["architecture"],
                    "variant": row["variant"],
                    "tracker": row["tracker"],
                    "region_mode": row["region_mode"],
                    "direction": direction,
                    "class": cls_name,
                    "predicted_count": pred_count,
                    "manual_count": gt_count,
                    "signed_error": pred_count - gt_count,
                    "abs_error": abs_error,
                    "pct_error": pct_error(pred_count, gt_count),
                    "abs_pct_error": abs_pct_error(pred_count, gt_count),
                    "count_accuracy": count_accuracy(pred_count, gt_count),
                    "min_duration_s": min_duration_s,
                    "max_duration_s": max_duration_s,
                    "min_avg_conf": min_avg_conf,
                }
                count_rows.append(count_row)
                exp_count_rows.append(count_row)

                tp, fp, fn, time_mae = greedy_time_match(
                    pred_c["enter_time_s"].tolist(),
                    gt_c["enter_time_s"].tolist(),
                    tol_s=match_tolerance_s,
                )
                precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
                event_row = {
                    "experiment_key": row["experiment_key"],
                    "plot_label": plot_label,
                    "video_key": row["video_key"],
                    "architecture": row["architecture"],
                    "variant": row["variant"],
                    "tracker": row["tracker"],
                    "region_mode": row["region_mode"],
                    "direction": direction,
                    "class": cls_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "matched_time_mae_s": time_mae,
                    "match_tolerance_s": match_tolerance_s,
                    "min_duration_s": min_duration_s,
                    "max_duration_s": max_duration_s,
                    "min_avg_conf": min_avg_conf,
                }
                event_rows.append(event_row)
                exp_event_rows.append(event_row)

        exp_count_df = pd.DataFrame(exp_count_rows)
        exp_event_df = pd.DataFrame(exp_event_rows)

        for direction in DIRECTIONS:
            subset = exp_count_df[exp_count_df["direction"] == direction]
            manual_total = int(subset["manual_count"].sum())
            abs_error_total = int(subset["abs_error"].sum())
            overall_wape = abs_error_total / manual_total if manual_total > 0 else np.nan
            macro_abs_pct_error = float(subset["abs_pct_error"].mean()) if not subset.empty else np.nan
            f1_subset = exp_event_df[exp_event_df["direction"] == direction]
            macro_f1 = float(f1_subset["f1"].mean()) if not f1_subset.empty else np.nan
            keke_row = f1_subset[f1_subset["class"] == "keke"]
            keke_f1 = float(keke_row["f1"].iloc[0]) if not keke_row.empty else np.nan
            summary_rows.append(
                {
                    "experiment_key": row["experiment_key"],
                    "plot_label": plot_label,
                    "video_key": row["video_key"],
                    "scene": row.get("scene"),
                    "architecture": row["architecture"],
                    "variant": row["variant"],
                    "tracker": row["tracker"],
                    "region_mode": row["region_mode"],
                    "direction": direction,
                    "manual_total": manual_total,
                    "abs_error_total": abs_error_total,
                    "overall_wape": overall_wape,
                    "macro_abs_pct_error": macro_abs_pct_error,
                    "macro_event_f1": macro_f1,
                    "keke_f1": keke_f1,
                    "processing_fps": row.get("processing_fps"),
                }
            )

    count_df = pd.DataFrame(count_rows)
    event_df = pd.DataFrame(event_rows)
    summary_df = pd.DataFrame(summary_rows)
    return count_df, event_df, summary_df



def compute_sensitivity(
    suite_index: pd.DataFrame,
    conf_grid: Sequence[float],
    duration_grid: Sequence[float],
    max_duration_s: float,
    match_tolerance_s: float,
) -> pd.DataFrame:
    baseline_cache: Dict[str, pd.DataFrame] = {}
    rows: List[dict] = []

    for _, row in suite_index.iterrows():
        segments = load_segments(row)
        baseline = load_baseline(row["baseline_path"], baseline_cache)
        plot_label = build_plot_label(row)

        for min_avg_conf in conf_grid:
            for min_duration_s in duration_grid:
                filtered = filter_segments(
                    segments,
                    min_duration_s=min_duration_s,
                    max_duration_s=max_duration_s,
                    min_avg_conf=min_avg_conf,
                )

                for direction in DIRECTIONS:
                    pred_d = filtered[filtered["direction"] == direction]
                    gt_d = baseline[baseline["direction"] == direction]
                    manual_total = 0
                    abs_error_total = 0
                    keke_f1 = np.nan

                    for cls_name in DOMINANT_CLASSES:
                        pred_c = pred_d[pred_d["class"] == cls_name]
                        gt_c = gt_d[gt_d["class"] == cls_name]
                        pred_count = int(len(pred_c))
                        gt_count = int(len(gt_c))
                        manual_total += gt_count
                        abs_error_total += abs(pred_count - gt_count)

                        if cls_name == "keke":
                            tp, fp, fn, _ = greedy_time_match(
                                pred_c["enter_time_s"].tolist(),
                                gt_c["enter_time_s"].tolist(),
                                tol_s=match_tolerance_s,
                            )
                            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                            keke_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

                    rows.append(
                        {
                            "experiment_key": row["experiment_key"],
                            "plot_label": plot_label,
                            "video_key": row["video_key"],
                            "direction": direction,
                            "min_avg_conf": float(min_avg_conf),
                            "min_duration_s": float(min_duration_s),
                            "max_duration_s": float(max_duration_s),
                            "overall_wape": abs_error_total / manual_total if manual_total > 0 else np.nan,
                            "keke_f1": keke_f1,
                        }
                    )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def choose_focus_experiments(summary_df: pd.DataFrame) -> List[str]:
    desired = []
    for architecture in ("YOLO11l", "YOLO26l"):
        desired.extend(
            [
                f"{architecture} default | botsort | ROI",
                f"{architecture} tuned | botsort | ROI",
                f"{architecture} tuned | bytetrack | ROI",
                f"{architecture} tuned | botsort | Full frame",
            ]
        )
    available = summary_df["plot_label"].drop_duplicates().tolist()
    selected = [x for x in desired if x in available]
    if selected:
        return selected
    return available[: min(8, len(available))]



def save_overall_wape_heatmap(summary_df: pd.DataFrame, outdir: Path) -> None:
    pivot = summary_df.copy()
    pivot["scene_direction"] = pivot["video_key"] + "\n" + pivot["direction"]
    heat = pivot.pivot(index="plot_label", columns="scene_direction", values="overall_wape")
    heat = heat.sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * heat.shape[1]), max(5, 0.45 * heat.shape[0] + 1)))
    data = heat.values.astype(float)
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Overall count WAPE by experiment, video, and direction")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="WAPE")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_overall_wape_heatmap.png", dpi=250)
    plt.close(fig)


def save_overall_wape_heatmap_split_arch(summary_df: pd.DataFrame, outdir: Path) -> None:
    """
    Save a side-by-side WAPE heatmap split by detector family:
      - left: YOLO11l experiments
      - right: YOLO26l experiments

    Each panel is intended to be 8 x 4 for the full_matrix setting:
      rows   = experiment variants within that architecture
      cols   = [mubi_downward, mubi_upward, yola_downward, yola_upward] (or sorted equivalent)
    """
    plot_df = summary_df.copy()
    plot_df["scene_direction"] = plot_df["video_key"] + "\n" + plot_df["direction"]

    # Keep a stable, paper-friendly column order if present
    preferred_cols = [
        "yola_road\ndownward",
        "yola_road\nupward",
        "mubi_road\ndownward",
        "mubi_road\nupward",
    ]
    # plot_df["scene_direction"] = (
    # plot_df["video_key"]
    #     .replace({
    #         "yola_road": "Yola Road",
    #         "mubi_road": "Mubi Road",
    #     })
    #     + "\n"
    #     + plot_df["direction"].replace({
    #         "downward": "Downward",
    #         "upward": "Upward",
    #     })
    # )

    # # Keep a stable, paper-friendly column order if present
    # preferred_cols = [
    #     "Yola Road\nDownward",
    #     "Yola Road\nUpward",
    #     "Mubi Road\nDownward",
    #     "Mubi Road\nUpward",
    # ]
    existing_cols = [c for c in preferred_cols if c in set(plot_df["scene_direction"])]
    if not existing_cols:
        existing_cols = sorted(plot_df["scene_direction"].dropna().unique().tolist())

    # Build one heatmap per architecture
    architectures = ["YOLO11l", "YOLO26l"]
    heatmaps = {}

    for arch in architectures:
        sub = plot_df[plot_df["architecture"] == arch].copy()
        if sub.empty:
            continue

        heat = sub.pivot(index="plot_label", columns="scene_direction", values="overall_wape")

        # Reorder columns
        heat = heat.reindex(columns=existing_cols)

        print(heat.columns)

        # Cleaner row labels within each architecture:
        # remove repeated "YOLO11l " / "YOLO26l " prefix
        clean_index = []
        for idx in heat.index.tolist():
            label = str(idx)
            prefix = f"{arch} "
            if label.startswith(prefix):
                label = label[len(prefix):]
            label = label.replace("Full frame", "Full")
            clean_index.append(label)
        heat.index = clean_index

        # Stable row order: default before tuned, botsort before bytetrack, ROI before Full (Full frame)
        def row_sort_key(label: str):
            s = label.lower()
            variant_rank = 0 if "default" in s else 1
            tracker_rank = 0 if "botsort" in s else 1
            region_rank = 0 if "roi" in s else 1
            return (variant_rank, tracker_rank, region_rank, s)

        ordered_rows = sorted(heat.index.tolist(), key=row_sort_key)
        heat = heat.reindex(index=ordered_rows)

        heatmaps[arch] = heat

    # If only one architecture is available, still plot robustly
    available_arches = [a for a in architectures if a in heatmaps]
    if not available_arches:
        raise ValueError("No architecture-specific data found for WAPE heatmap.")

    # Shared color scale across both panels
    all_vals = np.concatenate(
        [heatmaps[a].to_numpy(dtype=float).ravel() for a in available_arches]
    )
    finite_vals = all_vals[np.isfinite(all_vals)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(
        1,
        len(available_arches),
        # figsize=(14, max(5.5, 0.55 * max(h.shape[0] for h in heatmaps.values()) + 1.5)),
        figsize=(12, 6.5),
        sharey=True,
    )

    if len(available_arches) == 1:
        axes = [axes]

    last_im = None
    for ax, arch in zip(axes, available_arches):
        heat = heatmaps[arch]
        data = heat.to_numpy(dtype=float)

        last_im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=0)

        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)
        
        ax.set_title(arch)
        # ax.set_xlabel("Video and direction")

        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    # ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
                    text = ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontweight='bold')
                    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    axes[0].set_ylabel("Experiment setting")
    # fig.suptitle("Overall count WAPE by experiment, split by detector family", y=0.98)

    # Reserve space on the right for a dedicated colorbar axis
    # fig.subplots_adjust(left=0.24, right=0.90, top=0.90, bottom=0.12, wspace=0.18)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.94, bottom=0.10, wspace=0.06)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("WAPE")

    fig.savefig(outdir / "fig1_overall_wape_heatmap_split_arch.png", dpi=250)
    fig.savefig(outdir / "overall_wape_heatmap_split_arch.png", dpi=250)
    plt.close(fig)

def save_count_grid_grouped_bars(count_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str]) -> None:
    """
    2x2 grouped bar chart:
      - one panel per video + direction
      - x-axis = dominant vehicle classes
      - bars = manual baseline + selected counting pipeline configurations

    Recommended for the paper with focus_experiments (8 pipelines).
    If you pass all experiments from full_matrix, the figure will be much denser.
    """
    subset = count_df[count_df["plot_label"].isin(focus_experiments)].copy()
    if subset.empty:
        return

    subset["scene_direction"] = subset["video_key"] + "\n" + subset["direction"]

    preferred_panels = [
        "yola_road\ndownward",
        "yola_road\nupward",
        "mubi_road\ndownward",
        "mubi_road\nupward",
    ]
    panels = [p for p in preferred_panels if p in set(subset["scene_direction"])]
    if not panels:
        panels = sorted(subset["scene_direction"].dropna().unique().tolist())

    exps = [x for x in focus_experiments if x in set(subset["plot_label"])]
    if not exps:
        return

    def short_label(label: str) -> str:
        s = str(label)
        s = s.replace("YOLO11l", "11L")
        s = s.replace("YOLO26l", "26L")
        s = s.replace("default", "def")
        s = s.replace("tuned", "tuned")
        s = s.replace("botsort", "BoT")
        s = s.replace("bytetrack", "Byte")
        s = s.replace("Full frame", "Full")
        return s

    x = np.arange(len(DOMINANT_CLASSES))
    n_series = len(exps) + 1  # +1 for manual baseline
    width = 0.82 / n_series

    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(exps))))[: len(exps)]

    # fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        panel_df = subset[subset["scene_direction"] == panel].copy()

        # Manual baseline counts (same for all experiments within a panel)
        baseline_vals = (
            panel_df[["class", "manual_count"]]
            .drop_duplicates()
            .set_index("class")
            .reindex(DOMINANT_CLASSES)["manual_count"]
            .fillna(0)
            .to_numpy()
        )

        # baseline bar first
        ax.bar(
            x - (n_series - 1) * width / 2.0,
            baseline_vals,
            width=width,
            label="Manual baseline",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
        )

        # pipeline bars
        for idx, exp in enumerate(exps):
            vals = (
                panel_df[panel_df["plot_label"] == exp]
                .set_index("class")
                .reindex(DOMINANT_CLASSES)["predicted_count"]
                .fillna(0)
                .to_numpy()
            )

            ax.bar(
                x - (n_series - 1) * width / 2.0 + (idx + 1) * width,
                vals,
                width=width,
                label=short_label(exp),
                alpha=0.9,
                color=colors[idx],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(DOMINANT_CLASSES)
        ax.set_title(panel.replace("_road", "").replace("\n", " | "))
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    # hide unused axes if fewer than 4 panels
    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(labels)), # ncol=min(3, len(labels)),
        bbox_to_anchor=(0.5, 0.95), # bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )

    # fig.suptitle("Manual baseline vs counting pipeline counts by class", y=0.99)
    # fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Manual baseline vs counting pipeline counts by class", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    fig.savefig(outdir / "fig7_count_grid_grouped_bars.png", dpi=250)
    fig.savefig(outdir / "count_grid_grouped_bars.png", dpi=250)
    plt.close(fig)


def save_count_grid_grouped_bars_selected(count_df: pd.DataFrame, outdir: Path, selected_experiments: Sequence[str]) -> None:
    """
    2x2 grouped bar chart:
      - one panel per video + direction
      - x-axis = dominant vehicle classes
      - bars = manual baseline + selected counting pipeline configurations

    Recommended for the paper with focus_experiments (8 pipelines).
    If you pass all experiments from full_matrix, the figure will be much denser.
    """

    selected_experiments = ['YOLO11l tuned | botsort | ROI', 'YOLO11l tuned | bytetrack | ROI', 'YOLO26l tuned | botsort | ROI', 'YOLO26l tuned | bytetrack | ROI']

    subset = count_df[count_df["plot_label"].isin(selected_experiments)].copy()
    if subset.empty:
        return

    subset["scene_direction"] = subset["video_key"] + "\n" + subset["direction"]

    preferred_panels = [
        "yola_road\ndownward",
        "yola_road\nupward",
        "mubi_road\ndownward",
        "mubi_road\nupward",
    ]
    panels = [p for p in preferred_panels if p in set(subset["scene_direction"])]
    if not panels:
        panels = sorted(subset["scene_direction"].dropna().unique().tolist())

    exps = [x for x in selected_experiments if x in set(subset["plot_label"])]
    if not exps:
        return

    def short_label(label: str) -> str:
        s = str(label)
        s = s.replace("YOLO11l", "11L")
        s = s.replace("YOLO26l", "26L")
        s = s.replace("default", "def")
        s = s.replace("tuned", "tuned")
        s = s.replace("botsort", "BoT")
        s = s.replace("bytetrack", "Byte")
        s = s.replace("Full frame", "Full")
        return s

    x = np.arange(len(DOMINANT_CLASSES))
    n_series = len(exps) + 1  # +1 for manual baseline
    width = 0.82 / n_series

    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(exps))))[: len(exps)]

    # fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=True)
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        panel_df = subset[subset["scene_direction"] == panel].copy()

        # Manual baseline counts (same for all experiments within a panel)
        baseline_vals = (
            panel_df[["class", "manual_count"]]
            .drop_duplicates()
            .set_index("class")
            .reindex(DOMINANT_CLASSES)["manual_count"]
            .fillna(0)
            .to_numpy()
        )

        # baseline bar first
        baseline_container = ax.bar(
            x - (n_series - 1) * width / 2.0,
            baseline_vals,
            width=width,
            label="Manual baseline",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
        )

        ax.bar_label(
            baseline_container,
            labels=[f"{int(v)}" if np.isfinite(v) else "" for v in baseline_vals],
            padding=2,
            fontsize=7,
            fontweight="bold",
            rotation=0, # 90,
        )

        # pipeline bars
        for idx, exp in enumerate(exps):
            vals = (
                panel_df[panel_df["plot_label"] == exp]
                .set_index("class")
                .reindex(DOMINANT_CLASSES)["predicted_count"]
                .fillna(0)
                .to_numpy()
            )

            bar_container = ax.bar(
                x - (n_series - 1) * width / 2.0 + (idx + 1) * width,
                vals,
                width=width,
                label=short_label(exp),
                edgecolor="black",
                linewidth=1.0,
                alpha=0.9,
                # color=colors[idx],
            )

            ax.bar_label(
                bar_container,
                labels=[f"{int(v)}" if np.isfinite(v) else "" for v in vals],
                padding=2,
                fontsize=7,
                fontweight="bold",
                rotation=0, # 90,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(DOMINANT_CLASSES)
        # ax.set_title(panel.replace("_road", "").replace("\n", " | "))
        ax.set_title(panel.replace("yola_road", "Yola Road").replace("mubi_road", "Mubi Road").replace("ward", "ward traffic").replace("\n", " | "))
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 275)

    # hide unused axes if fewer than 4 panels
    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(labels)), # ncol=min(3, len(labels)),
        bbox_to_anchor=(0.5, 0.99), # bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )

    # fig.suptitle("Manual baseline vs counting pipeline counts by class", y=0.99)
    # fig.tight_layout(rect=[0, 0, 1, 0.93])
    # fig.suptitle("Manual baseline vs counting pipeline counts by class", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(outdir / "fig7_count_grid_grouped_bars_selected.png", dpi=250)
    fig.savefig(outdir / "count_grid_grouped_bars_selected.png", dpi=250)
    plt.close(fig)


def save_keke_counts(count_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str]) -> None:
    keke = count_df[(count_df["class"] == "keke") & (count_df["plot_label"].isin(focus_experiments))].copy()
    keke["scene_direction"] = keke["video_key"] + "\n" + keke["direction"]
    scenes = list(dict.fromkeys(keke["scene_direction"].tolist()))
    exps = [x for x in focus_experiments if x in set(keke["plot_label"])]
    x = np.arange(len(scenes))
    width = 0.8 / max(1, len(exps))

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * len(scenes)), 5))
    manual = (
        keke[["scene_direction", "manual_count"]]
        .drop_duplicates()
        .set_index("scene_direction")
        .reindex(scenes)["manual_count"]
        .fillna(0)
        .to_numpy()
    )
    ax.plot(x, manual, marker="o", linewidth=2, label="Manual baseline")

    for idx, exp in enumerate(exps):
        subset = (
            keke[keke["plot_label"] == exp]
            .set_index("scene_direction")
            .reindex(scenes)["predicted_count"]
            .fillna(0)
            .to_numpy()
        )
        ax.bar(x + (idx - (len(exps) - 1) / 2.0) * width, subset, width=width, label=exp)

    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Count")
    ax.set_title("Keke counts: manual baseline vs model predictions")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_keke_counts.png", dpi=250)
    plt.close(fig)



def save_directional_error_panels(count_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str]) -> None:
    subset = count_df[count_df["plot_label"].isin(focus_experiments)].copy()
    subset["scene_direction"] = subset["video_key"] + "\n" + subset["direction"]
    panels = list(dict.fromkeys(subset["scene_direction"].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        panel_df = subset[subset["scene_direction"] == panel]
        exp_order = [x for x in focus_experiments if x in set(panel_df["plot_label"])]
        x = np.arange(len(DOMINANT_CLASSES))
        width = 0.8 / max(1, len(exp_order))
        for idx, exp in enumerate(exp_order):
            vals = (
                panel_df[panel_df["plot_label"] == exp]
                .set_index("class")
                .reindex(DOMINANT_CLASSES)["signed_error"]
                .fillna(0)
                .to_numpy()
            )
            ax.bar(x + (idx - (len(exp_order) - 1) / 2.0) * width, vals, width=width, label=exp)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(DOMINANT_CLASSES)
        ax.set_title(panel)
        ax.set_ylabel("Signed count error")
        ax.grid(axis="y", alpha=0.3)

    if len(axes) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Direction-specific class count errors", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig3_directional_class_error.png", dpi=250)
    plt.close(fig)


def save_keke_f1(event_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str]) -> None:
    keke = event_df[(event_df["class"] == "keke") & (event_df["plot_label"].isin(focus_experiments))].copy()
    keke["scene_direction"] = keke["video_key"] + "\n" + keke["direction"]
    scenes = list(dict.fromkeys(keke["scene_direction"].tolist()))
    exps = [x for x in focus_experiments if x in set(keke["plot_label"])]
    x = np.arange(len(scenes))
    width = 0.8 / max(1, len(exps))

    fig, ax = plt.subplots(figsize=(max(9, 1.8 * len(scenes)), 5))
    for idx, exp in enumerate(exps):
        vals = (
            keke[keke["plot_label"] == exp]
            .set_index("scene_direction")
            .reindex(scenes)["f1"]
            .fillna(0)
            .to_numpy()
        )
        ax.bar(x + (idx - (len(exps) - 1) / 2.0) * width, vals, width=width, label=exp)

    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Event-level F1")
    ax.set_title("Keke event matching F1 (time-tolerant count validation)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_keke_event_f1.png", dpi=250)
    plt.close(fig)



def save_conf_sensitivity(sensitivity_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str], fixed_duration_s: float) -> None:
    subset = sensitivity_df[
        (sensitivity_df["plot_label"].isin(focus_experiments))
        & (np.isclose(sensitivity_df["min_duration_s"], fixed_duration_s))
    ].copy()
    subset["scene_direction"] = subset["video_key"] + "\n" + subset["direction"]
    panels = list(dict.fromkeys(subset["scene_direction"].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels):
        panel_df = subset[subset["scene_direction"] == panel]
        for exp in [x for x in focus_experiments if x in set(panel_df["plot_label"])]:
            curve = panel_df[panel_df["plot_label"] == exp].sort_values("min_avg_conf")
            ax.plot(curve["min_avg_conf"], curve["keke_f1"], marker="o", label=exp)
        ax.set_title(panel)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Minimum average in-region confidence")
        ax.set_ylabel("Keke F1")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Keke F1 sensitivity to confidence threshold (min duration = {fixed_duration_s:.2f} s)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "fig5_conf_sensitivity_keke_f1.png", dpi=250)
    plt.close(fig)



def save_duration_sensitivity(sensitivity_df: pd.DataFrame, outdir: Path, focus_experiments: Sequence[str], fixed_conf: float) -> None:
    subset = sensitivity_df[
        (sensitivity_df["plot_label"].isin(focus_experiments))
        & (np.isclose(sensitivity_df["min_avg_conf"], fixed_conf))
    ].copy()
    subset["scene_direction"] = subset["video_key"] + "\n" + subset["direction"]
    panels = list(dict.fromkeys(subset["scene_direction"].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels):
        panel_df = subset[subset["scene_direction"] == panel]
        for exp in [x for x in focus_experiments if x in set(panel_df["plot_label"])]:
            curve = panel_df[panel_df["plot_label"] == exp].sort_values("min_duration_s")
            ax.plot(curve["min_duration_s"], curve["overall_wape"], marker="o", label=exp)
        ax.set_title(panel)
        ax.set_xlabel("Minimum dwell time in region (s)")
        ax.set_ylabel("Overall WAPE")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Overall WAPE sensitivity to minimum dwell time (min avg conf = {fixed_conf:.2f})", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "fig6_duration_sensitivity_overall_wape.png", dpi=250)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean paper-ready counting comparison plots from an ablation suite.")
    parser.add_argument("--suite-dir", default="./results/ablation_suite", help="Root directory produced by run_counting_ablation_suite.py")
    parser.add_argument("--outdir", default=None, help="Directory to save figures and CSV tables. Default: <suite-dir>/paper_plots")
    parser.add_argument("--min-duration-s", type=float, default=0.15, help="Primary minimum dwell time for the main analysis.")
    parser.add_argument("--max-duration-s", type=float, default=10.0, help="Maximum dwell time for the main analysis.")
    parser.add_argument("--min-avg-conf", type=float, default=0.40, help="Primary minimum average in-region confidence.")
    parser.add_argument("--match-tolerance-s", type=float, default=1.0, help="Time tolerance for one-to-one event matching against manual counts.")
    parser.add_argument("--conf-grid", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60], help="Grid for confidence sensitivity plots.")
    parser.add_argument("--duration-grid", nargs="+", type=float, default=[0.00, 0.15, 0.30, 0.50], help="Grid for dwell-time sensitivity plots.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    outdir = Path(args.outdir) if args.outdir else suite_dir / "paper_plots"
    ensure_dir(outdir)

    suite_index = load_suite_index(suite_dir).copy()
    suite_index["suite_dir"] = str(suite_dir)

    count_df, event_df, summary_df = compute_metrics(
        suite_index,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        min_avg_conf=args.min_avg_conf,
        match_tolerance_s=args.match_tolerance_s,
    )
    sensitivity_df = compute_sensitivity(
        suite_index,
        conf_grid=args.conf_grid,
        duration_grid=args.duration_grid,
        max_duration_s=args.max_duration_s,
        match_tolerance_s=args.match_tolerance_s,
    )

    count_df.to_csv(outdir / "count_metrics.csv", index=False)
    event_df.to_csv(outdir / "event_metrics.csv", index=False)
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    sensitivity_df.to_csv(outdir / "threshold_sensitivity.csv", index=False)

    ranking = summary_df.sort_values(["video_key", "direction", "overall_wape", "keke_f1"], ascending=[True, True, True, False])
    ranking.to_csv(outdir / "ranked_experiments.csv", index=False)

    focus_experiments = choose_focus_experiments(summary_df)
    (outdir / "focus_experiments.json").write_text(json.dumps(focus_experiments, indent=2), encoding="utf-8")

    save_overall_wape_heatmap(summary_df, outdir)
    save_overall_wape_heatmap_split_arch(summary_df, outdir)
    
    # save_count_grid_grouped_bars(count_df, outdir, focus_experiments)
    save_count_grid_grouped_bars(count_df, outdir, summary_df["plot_label"].drop_duplicates().tolist()) # runs for all experiment

    # selected_experiments = ['YOLO11l default | botsort | ROI', 'YOLO11l default | bytetrack | ROI', 'YOLO11l tuned | botsort | ROI', 'YOLO11l tuned | bytetrack | ROI', 'YOLO26l default | botsort | ROI', 'YOLO26l default | bytetrack | ROI', 'YOLO26l tuned | botsort | ROI', 'YOLO26l tuned | bytetrack | ROI']
    selected_experiments = ['YOLO11l tuned | botsort | ROI', 'YOLO11l tuned | bytetrack | ROI', 'YOLO26l tuned | botsort | ROI', 'YOLO26l tuned | bytetrack | ROI']
    save_count_grid_grouped_bars_selected(count_df, outdir, selected_experiments) # for manually selected experiments

    save_keke_counts(count_df, outdir, focus_experiments)
    save_directional_error_panels(count_df, outdir, focus_experiments)
    save_keke_f1(event_df, outdir, focus_experiments)
    save_conf_sensitivity(sensitivity_df, outdir, focus_experiments, fixed_duration_s=args.min_duration_s)
    save_duration_sensitivity(sensitivity_df, outdir, focus_experiments, fixed_conf=args.min_avg_conf)

    print(f"Saved tables and figures under: {outdir}")


if __name__ == "__main__":
    main()
