import os
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def traffic_flow_plotter_combined(tuned,default,baseline,outdir,direction,min_duration_s=0.15,max_duration_s=10.0,min_avg_conf=0.4):

    color_map = {
        "bicycle": "tab:red",
        "bus": "tab:purple",
        "car": "tab:blue",
        "keke": "tab:orange",
        "motorcycle": "tab:pink",
        "truck": "tab:green",
        "van": "tab:brown",
    }

    classes = ("bus", "car", "keke", "truck")
    models = ("tuned", "default", "baseline")

    # accumulate counts for the bar plot
    counts = {cls: {"tuned": 0, "default": 0, "baseline": 0} for cls in classes}

    plt.figure(figsize=(12, 7))

    # Plot each class twice: tuned '-' and default ':'
    # for object_cls in ('bicycle', 'bus', 'car', 'keke', 'motorcycle', 'truck'):
    for object_cls in classes:
        tdf = tuned[tuned["class"] == object_cls]
        ddf = default[default["class"] == object_cls]
        bdf = baseline[baseline["class"] == object_cls]

        # filtering by duration to remove noisy/very long stationary objects
        tdf = tdf[(tdf["duration_s"] >= min_duration_s) & (tdf["duration_s"] <= max_duration_s)]
        ddf = ddf[(ddf["duration_s"] >= min_duration_s) & (ddf["duration_s"] <= max_duration_s)]

        # filtering by average confidence in region
        tdf = tdf[tdf["avg_conf_in_region"] >= min_avg_conf]
        ddf = ddf[ddf["avg_conf_in_region"] >= min_avg_conf]

        # # filtering by 0.15s durations i.e. appears in atleast 4 frames for @30 fps video, to remove noisy objects
        # # also filtered out stationary objects in region for more than 10 seconds 
        # tdf = tdf[(tdf["duration_s"] >= 0.15) & (tdf["duration_s"] <= 10)]
        # ddf = ddf[(ddf["duration_s"] >= 0.15) & (ddf["duration_s"] <= 10)]


        # # filtering by 0.4 average confidence in region
        # tdf = tdf[tdf["avg_conf_in_region"] >= 0.4]
        # ddf = ddf[ddf["avg_conf_in_region"] >= 0.4]

        # specifying direction of travel
        if direction == "upward":
            # upwards travel filter
            tdf = tdf[tdf["enter_cy"] > tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] > ddf["exit_cy"]]
        
        elif direction == "downward":
            # downwards travel filter
            tdf = tdf[tdf["enter_cy"] < tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] < ddf["exit_cy"]]
        else:
            # stationary vehicles in counting segment region
            tdf = tdf[tdf["enter_cy"] == tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] == ddf["exit_cy"]]

        # filter baseline by direction of travel        
        bdf = bdf[bdf["direction"] == direction]

        
        xt = np.sort(tdf["enter_time_s"].to_numpy())
        yt = np.arange(1, len(xt) + 1, dtype=float)
        xd = np.sort(ddf["enter_time_s"].to_numpy())
        yd = np.arange(1, len(xd) + 1, dtype=float)
        xb = np.sort(bdf["enter_time_s"].to_numpy())
        yb = np.arange(1, len(xb) + 1, dtype=float)

        # store counts for bar plot
        counts[object_cls]["tuned"] = len(xt)
        counts[object_cls]["default"] = len(xd)
        counts[object_cls]["baseline"] = len(xb)

        # building label
        tdf_label = object_cls + " - tuned (" + str(len(xt)) + ")"
        ddf_label = object_cls + " - default (" + str(len(xd)) + ")"
        bdf_label = object_cls + " - baseline (" + str(len(xb)) + ")"
        # print(f'{tdf_label}\n{ddf_label}\n{bdf_label}')

        plt.plot(xt, yt, linestyle="-", color=color_map[object_cls], label=tdf_label)
        plt.plot(xd, yd, linestyle=":", color=color_map[object_cls], label=ddf_label)
        plt.plot(xb, yb, linestyle="--", color=color_map[object_cls], label=bdf_label)


    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative count (segments/entries)")  
    plt.legend()
    plt.title(f"Vehicle count over time (by classes) - {direction} flow")
    plt.tight_layout()
    outpath=os.path.join(outdir, f"{direction}_traffic_flow_analysis.png")
    plt.savefig(outpath, dpi=200)


def traffic_flow_plotter(tuned, default, baseline, outdir, direction, min_duration_s=0.15, max_duration_s=10.0, min_avg_conf=0.4):
    """
    Creates:
      1) A 2x2 figure: one subplot per class, each subplot has 3 cumulative curves (tuned/default/baseline)
      2) A grouped bar chart figure showing distribution counts across classes for tuned/default/baseline

    Saves:
      - {direction}_traffic_flow_analysis_2x2.png
      - {direction}_traffic_flow_distribution.png
    """

    color_map = {
        "bicycle": "tab:red",
        "bus": "tab:purple",
        "car": "tab:blue",
        "keke": "tab:orange",
        "motorcycle": "tab:pink",
        "truck": "tab:green",
        "van": "tab:brown",
    }

    classes = ("bus", "car", "keke", "truck")
    models = ("tuned", "default", "baseline")

    # accumulate counts for the bar plot
    counts = {cls: {"tuned": 0, "default": 0, "baseline": 0} for cls in classes}

    # ---- 2x2 cumulative plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=False)
    axes = axes.ravel()

    # consistent per-model styling across subplots
    model_style = {
        "tuned": "-",
        "default": ":",
        "baseline": "--",
    }

    for ax, object_cls in zip(axes, classes):
        tdf = tuned[tuned["class"] == object_cls]
        ddf = default[default["class"] == object_cls]
        bdf = baseline[baseline["class"] == object_cls]

        # duration filter (noisy + long stationary)
        tdf = tdf[(tdf["duration_s"] >= min_duration_s) & (tdf["duration_s"] <= max_duration_s)]
        ddf = ddf[(ddf["duration_s"] >= min_duration_s) & (ddf["duration_s"] <= max_duration_s)]

        # confidence filter
        tdf = tdf[tdf["avg_conf_in_region"] >= min_avg_conf]
        ddf = ddf[ddf["avg_conf_in_region"] >= min_avg_conf]

        # direction filter (tuned/default)
        if direction == "upward":
            tdf = tdf[tdf["enter_cy"] > tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] > ddf["exit_cy"]]
        elif direction == "downward":
            tdf = tdf[tdf["enter_cy"] < tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] < ddf["exit_cy"]]
        else:
            tdf = tdf[tdf["enter_cy"] == tdf["exit_cy"]]
            ddf = ddf[ddf["enter_cy"] == ddf["exit_cy"]]

        # baseline direction filter
        bdf = bdf[bdf["direction"] == direction]

        # cumulative series
        xt = np.sort(tdf["enter_time_s"].to_numpy())
        yt = np.arange(1, len(xt) + 1, dtype=float)

        xd = np.sort(ddf["enter_time_s"].to_numpy())
        yd = np.arange(1, len(xd) + 1, dtype=float)

        xb = np.sort(bdf["enter_time_s"].to_numpy())
        yb = np.arange(1, len(xb) + 1, dtype=float)

        # store counts for bar plot
        counts[object_cls]["tuned"] = len(xt)
        counts[object_cls]["default"] = len(xd)
        counts[object_cls]["baseline"] = len(xb)

        # labels (only show model in legend; count goes in subplot title)
        tuned_label = "tuned"
        default_label = "default"
        baseline_label = "baseline"

        ax.plot(xt, yt, linestyle=model_style["tuned"], color=color_map[object_cls], label=tuned_label)
        ax.plot(xd, yd, linestyle=model_style["default"], color=color_map[object_cls], label=default_label)
        ax.plot(xb, yb, linestyle=model_style["baseline"], color=color_map[object_cls], label=baseline_label)

        ax.set_title(f"{object_cls}  (tuned={len(xt)}, default={len(xd)}, baseline={len(xb)})")        
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative count")

        # keep legend inside each subplot (3 items only, so readable)
        ax.legend(loc="upper left", title="Model", fontsize=9)

        print(f"{object_cls}  (tuned={len(xt)}, default={len(xd)}, baseline={len(xb)})")

    fig.suptitle(f"Vehicle count over time (2x2 by class) - {direction} flow", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(outdir, f"{direction}_traffic_flow_analysis_2x2.png")
    fig.savefig(outpath, dpi=200)

    # ---- Grouped bar chart for distribution (unchanged idea) ----
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    x = np.arange(len(classes))
    width = 0.25

    tuned_vals = [counts[c]["tuned"] for c in classes]
    default_vals = [counts[c]["default"] for c in classes]
    baseline_vals = [counts[c]["baseline"] for c in classes]

    ax2.bar(x - width, tuned_vals, width, label="tuned")
    ax2.bar(x, default_vals, width, label="default")
    ax2.bar(x + width, baseline_vals, width, label="baseline")

    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_xlabel("Vehicle class")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Traffic count distribution - {direction} flow")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(title="Model")

    for container in ax2.containers:
        ax2.bar_label(container, padding=2, fontsize=9, fontweight="bold")

    ymax = max(tuned_vals + default_vals + baseline_vals) if (tuned_vals + default_vals + baseline_vals) else 0
    ax2.set_ylim(0, ymax + max(1, int(0.15 * ymax)))

    fig2.tight_layout()
    bar_outpath = os.path.join(outdir, f"{direction}_traffic_flow_distribution.png")
    fig2.savefig(bar_outpath, dpi=200)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot tuned/default/baseline traffic flow metrics from CSV files."
    )
    parser.add_argument(
        "--tuned-segments",
        default="./results/tuned_region_segments.csv",
        help="Path to tuned model region segments CSV.",
    )
    parser.add_argument(
        "--default-segments",
        default="./results/default_region_segments.csv",
        help="Path to default model region segments CSV.",
    )
    parser.add_argument(
        "--baseline-segments",
        default="./baseline/yola_road_mp4_baseline_vehicle_count.csv",
        help="Path to baseline/manual count CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="./results/plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["upward", "downward"],
        choices=["upward", "downward", "stationary"],
        help="One or more directions to plot.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=0.15,
        help="Minimum segment duration to keep.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=10.0,
        help="Maximum segment duration to keep.",
    )
    parser.add_argument(
        "--min-avg-conf",
        type=float,
        default=0.4,
        help="Minimum average in-region confidence to keep.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip the combined per-direction line plot.",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip the 2x2 cumulative plot and grouped bar chart.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively with plt.show().",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tuned = pd.read_csv(args.tuned_segments)
    default = pd.read_csv(args.default_segments)
    baseline = pd.read_csv(args.baseline_segments)

    for direction in args.directions:
        print(f"-------------{direction} traffic flow-------------")

        if not args.skip_combined:
            traffic_flow_plotter_combined(
                tuned,
                default,
                baseline,
                args.outdir,
                direction=direction,
                min_duration_s=args.min_duration_s,
                max_duration_s=args.max_duration_s,
                min_avg_conf=args.min_avg_conf,
            )

        if not args.skip_grid:
            traffic_flow_plotter(
                tuned,
                default,
                baseline,
                args.outdir,
                direction=direction,
                min_duration_s=args.min_duration_s,
                max_duration_s=args.max_duration_s,
                min_avg_conf=args.min_avg_conf,
            )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
    
#     tuned_segments_path = "./results/tuned_region_segments.csv"
#     default_segments_path = "./results/default_region_segments.csv"
#     baseline_segments_path = "./baseline/yola_road_mp4_baseline_vehicle_count.csv"
#     tuned_per_frame_path = "./results/tuned_per_frame.csv"
#     default_per_frame_path = "./results/default_per_frame.csv"
#     outdir = "./results/plots"

#     os.makedirs(outdir, exist_ok=True)

#     tuned = pd.read_csv(tuned_segments_path)
#     default = pd.read_csv(default_segments_path)
#     baseline = pd.read_csv(baseline_segments_path)

#     # upward direction
#     print('-------------upward traffic flow-------------')
#     traffic_flow_plotter_combined(tuned,default,baseline,outdir,direction="upward")
#     traffic_flow_plotter(tuned,default,baseline,outdir,direction="upward")

#     # downward direction
#     print('-------------downward traffic flow-------------')
#     traffic_flow_plotter(tuned,default,baseline,outdir,direction="downward")    
#     traffic_flow_plotter_combined(tuned,default,baseline,outdir,direction="downward")    
#     plt.show()
#     # plt.close()
