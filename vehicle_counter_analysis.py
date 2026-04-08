import argparse
import cv2
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# -------------------------
# Defaults
# -------------------------
DEFAULT_VIDEO_PATH = "./datasets/test/yola_road.mp4"
DEFAULT_REGION_POINTS = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

DEFAULT_TUNED_MODEL = "runs/detect/train/weights/best.pt"
DEFAULT_DEFAULT_MODEL = "yolo11l.pt"

# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
DEFAULT_COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]

DEFAULT_TRACKER_CFG = "botsort.yaml" # or bytetrack.yaml
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45
DEFAULT_OUT_DIR = "./results"


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def region_contour(points: List[Tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def in_region(contour: np.ndarray, cx: float, cy: float) -> bool:
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def xyxy_centroid(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def parse_region_points(value: str) -> List[Tuple[int, int]]:
    """
    Expected format:
      "20,400 1080,400 1080,360 20,360"
    """
    points = []
    for token in value.strip().split():
        x_str, y_str = token.split(",")
        points.append((int(x_str), int(y_str)))
    if len(points) < 3:
        raise argparse.ArgumentTypeError("Region must contain at least 3 points.")
    return points


def parse_class_filter(value: str) -> Optional[List[int]]:
    """
    Examples:
      "none" -> None
      "2,3,5,7" -> [2, 3, 5, 7]
      "" -> None
    """
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"", "none", "null"}:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_output_paths(out_dir: str, model_label: str, save_video: bool, export_per_frame_csv: bool):
    video_out = os.path.join(out_dir, f"annotated_{model_label}.mp4") if save_video else None
    per_frame_csv = os.path.join(out_dir, f"{model_label}_per_frame.csv") if export_per_frame_csv else None
    segments_csv = os.path.join(out_dir, f"{model_label}_region_segments.csv")
    summary_json = os.path.join(out_dir, f"{model_label}_summary.json")
    return video_out, per_frame_csv, segments_csv, summary_json


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file: {video_path}"
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not determine FPS from video: {video_path}")
    return fps


@dataclass
class TrackState:
    cls_name: str
    ever_entered: bool = False
    inside: bool = False
    enter_time: Optional[float] = None
    enter_frame: Optional[int] = None

    enter_cx: Optional[float] = None
    enter_cy: Optional[float] = None
    last_cx: Optional[float] = None
    last_cy: Optional[float] = None

    conf_sum_inside: float = 0.0
    conf_max_inside: float = 0.0
    frames_inside: int = 0
    segments: List[dict] = field(default_factory=list)

    def start_segment(self, t: float, frame_idx: int, cx: float, cy: float):
        self.inside = True
        self.ever_entered = True
        self.enter_time = t
        self.enter_frame = frame_idx

        self.enter_cx = cx
        self.enter_cy = cy
        self.last_cx = cx
        self.last_cy = cy

        self.conf_sum_inside = 0.0
        self.conf_max_inside = 0.0
        self.frames_inside = 0

    def update_inside(self, conf: float, cx: float, cy: float):
        self.frames_inside += 1
        self.conf_sum_inside += conf
        self.conf_max_inside = max(self.conf_max_inside, conf)
        self.last_cx = cx
        self.last_cy = cy

    def end_segment(self, t_exit: float, frame_exit: int, track_id: int):
        if self.enter_time is None or self.enter_frame is None:
            return

        duration = t_exit - self.enter_time
        avg_conf = (self.conf_sum_inside / self.frames_inside) if self.frames_inside > 0 else 0.0

        self.segments.append({
            "track_id": track_id,
            "class": self.cls_name,
            "enter_time_s": self.enter_time,
            "exit_time_s": t_exit,
            "duration_s": duration,
            "enter_frame": self.enter_frame,
            "exit_frame": frame_exit,
            "enter_cx": self.enter_cx,
            "enter_cy": self.enter_cy,
            "exit_cx": self.last_cx,
            "exit_cy": self.last_cy,
            "avg_conf_in_region": avg_conf,
            "max_conf_in_region": self.conf_max_inside,
            "frames_in_region": self.frames_inside,
        })

        self.inside = False
        self.enter_time = None
        self.enter_frame = None


def run_video_for_model(
    *,
    video_path: str,
    model_path: str,
    model_label: str,
    contour: np.ndarray,
    output_video_path: Optional[str],
    per_frame_csv_path: Optional[str],
    segments_csv_path: str,
    summary_json_path: str,
    fps: float,
    tracker_cfg: str,
    conf_thres: float,
    iou_thres: float,
    out_dir: str,
    save_annotated_video: bool,
    export_per_frame_csv: bool,
    vehicle_class_filter: Optional[List[int]] = None,
):
    ensure_dir(out_dir)

    model = YOLO(model_path)
    names = model.names

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path and save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    track_states: Dict[int, TrackState] = {}
    per_frame_rows = []
    all_segments = []

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps

        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )
        r0 = results[0]
        boxes = r0.boxes

        annotated = r0.plot()
        cv2.polylines(annotated, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            tids = boxes.id.cpu().numpy().astype(int)

            for bb, conf, cls_id, tid in zip(xyxy, confs, clss, tids):
                if vehicle_class_filter is not None and cls_id not in vehicle_class_filter:
                    continue

                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                x1, y1, x2, y2 = bb
                cx, cy = xyxy_centroid(bb)
                inside = in_region(contour, cx, cy)

                if tid not in track_states:
                    track_states[tid] = TrackState(cls_name=cls_name)

                st = track_states[tid]

                if inside and not st.inside:
                    st.start_segment(timestamp_s, frame_idx, cx, cy)

                if inside and st.inside:
                    st.update_inside(float(conf), cx, cy)

                if (not inside) and st.inside:
                    st.end_segment(timestamp_s, frame_idx, tid)

                if export_per_frame_csv:
                    per_frame_rows.append({
                        "model": model_label,
                        "timestamp_s": timestamp_s,
                        "frame": frame_idx,
                        "track_id": tid,
                        "class": cls_name,
                        "conf": float(conf),
                        "in_region": bool(inside),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cx": float(cx),
                        "cy": float(cy),
                    })

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1

    end_time_s = (frame_idx - 1) / fps if frame_idx > 0 else 0.0
    for tid, st in track_states.items():
        if st.inside:
            st.end_segment(end_time_s, frame_idx - 1, tid)
        all_segments.extend(st.segments)

    t1 = time.time()
    elapsed = max(t1 - t0, 1e-9)
    processing_fps = frame_idx / elapsed

    cap.release()
    if writer is not None:
        writer.release()

    if export_per_frame_csv and per_frame_csv_path is not None:
        pd.DataFrame(per_frame_rows).to_csv(per_frame_csv_path, index=False)

    seg_df = pd.DataFrame(all_segments)
    if seg_df.empty:
        seg_df = pd.DataFrame(columns=[
            "model", "track_id", "class", "enter_time_s", "exit_time_s", "duration_s",
            "enter_frame", "exit_frame", "enter_cx", "enter_cy", "exit_cx", "exit_cy",
            "avg_conf_in_region", "max_conf_in_region", "frames_in_region"
        ])
    else:
        seg_df.insert(0, "model", model_label)

    seg_df.to_csv(segments_csv_path, index=False)

    total_segments = len(seg_df)
    total_unique_tracks = seg_df["track_id"].nunique() if total_segments else 0
    total_dwell = float(seg_df["duration_s"].sum()) if total_segments else 0.0
    avg_dwell = float(seg_df["duration_s"].mean()) if total_segments else 0.0
    avg_conf = float(seg_df["avg_conf_in_region"].mean()) if total_segments else 0.0

    summary = {
        "model": model_label,
        "model_path": model_path,
        "video_path": video_path,
        "tracker_cfg": tracker_cfg,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "frames_processed": frame_idx,
        "video_fps": fps,
        "processing_seconds": elapsed,
        "processing_fps": processing_fps,
        "segments": total_segments,
        "unique_tracks_with_region_segments": total_unique_tracks,
        "total_dwell_time_s": total_dwell,
        "avg_dwell_time_s": avg_dwell,
        "avg_conf_in_region": avg_conf,
        "vehicle_class_filter": vehicle_class_filter,
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return seg_df, summary


def plot_comparison(seg_a: pd.DataFrame, seg_b: pd.DataFrame, label_a: str, label_b: str, out_prefix: str):
    def counts_over_time(seg: pd.DataFrame):
        if seg.empty:
            return pd.DataFrame(columns=["t", "count"])
        t = np.floor(seg["enter_time_s"].values).astype(int)
        s = pd.Series(1, index=t).groupby(level=0).sum().sort_index()
        return pd.DataFrame({"t": s.index.values, "count": s.values})

    c_a = counts_over_time(seg_a)
    c_b = counts_over_time(seg_b)

    plt.figure()
    if not c_a.empty:
        plt.plot(c_a["t"], c_a["count"], label=label_a)
    if not c_b.empty:
        plt.plot(c_b["t"], c_b["count"], label=label_b)
    plt.xlabel("Time (s, binned)")
    plt.ylabel("Region entries (segments started)")
    plt.title("Region entry counts over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_counts_over_time.png", dpi=200)
    plt.close()

    plt.figure()
    if not seg_a.empty:
        plt.hist(seg_a["duration_s"], bins=30, alpha=0.6, label=label_a)
    if not seg_b.empty:
        plt.hist(seg_b["duration_s"], bins=30, alpha=0.6, label=label_b)
    plt.xlabel("Duration in region (s)")
    plt.ylabel("Segments")
    plt.title("Dwell time distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_dwell_hist.png", dpi=200)
    plt.close()

    plt.figure()
    if not seg_a.empty:
        plt.hist(seg_a["avg_conf_in_region"], bins=30, alpha=0.6, label=label_a)
    if not seg_b.empty:
        plt.hist(seg_b["avg_conf_in_region"], bins=30, alpha=0.6, label=label_b)
    plt.xlabel("Avg confidence in region")
    plt.ylabel("Segments")
    plt.title("Confidence distribution (avg in-region)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_conf_hist.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ROI-based vehicle tracking/counting for tuned and/or default YOLO models."
    )

    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Path to input video.")
    parser.add_argument(
        "--region-points",
        type=parse_region_points,
        default=DEFAULT_REGION_POINTS,
        help='ROI polygon points as "x1,y1 x2,y2 x3,y3 ...".',
    )

    parser.add_argument("--tuned-model", default=DEFAULT_TUNED_MODEL, help="Path to tuned model weights.")
    parser.add_argument("--default-model", default=DEFAULT_DEFAULT_MODEL, help="Path to baseline/default model weights.")
    parser.add_argument("--tuned-label", default="tuned", help="Output label for tuned model.")
    parser.add_argument("--default-label", default="default", help="Output label for default model.")

    parser.add_argument("--tracker", default=DEFAULT_TRACKER_CFG, help="Tracker config, e.g. botsort.yaml or bytetrack.yaml.")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES, help="Detection confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_IOU_THRES, help="Detection IoU threshold.")
    parser.add_argument("--fps", type=float, default=None, help="Override video FPS. By default it is read from the video file.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")

    parser.add_argument(
        "--tuned-class-filter",
        type=parse_class_filter,
        default=None,
        help='Comma-separated class IDs for tuned model, or "none".',
    )
    parser.add_argument(
        "--default-class-filter",
        type=parse_class_filter,
        default=DEFAULT_COCO_VEHICLE_CLASS_IDS,
        help='Comma-separated class IDs for default model, or "none". Default is "2,3,5,7".',
    )

    parser.add_argument("--skip-tuned", action="store_true", help="Skip tuned model run.")
    parser.add_argument("--skip-default", action="store_true", help="Skip default model run.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip comparison plots.")
    parser.add_argument("--save-annotated-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-per-frame-csv", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    fps = args.fps if args.fps is not None else get_video_fps(args.video_path)
    contour = region_contour(args.region_points)

    ran = {}

    if not args.skip_tuned:
        tuned_video_out, tuned_per_frame, tuned_segments, tuned_summary = build_output_paths(
            args.out_dir,
            args.tuned_label,
            args.save_annotated_video,
            args.export_per_frame_csv,
        )

        seg_tuned, sum_tuned = run_video_for_model(
            video_path=args.video_path,
            model_path=args.tuned_model,
            model_label=args.tuned_label,
            contour=contour,
            output_video_path=tuned_video_out,
            per_frame_csv_path=tuned_per_frame,
            segments_csv_path=tuned_segments,
            summary_json_path=tuned_summary,
            fps=fps,
            tracker_cfg=args.tracker,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            out_dir=args.out_dir,
            save_annotated_video=args.save_annotated_video,
            export_per_frame_csv=args.export_per_frame_csv,
            vehicle_class_filter=args.tuned_class_filter,
        )
        ran["tuned"] = (seg_tuned, sum_tuned)

    if not args.skip_default:
        default_video_out, default_per_frame, default_segments, default_summary = build_output_paths(
            args.out_dir,
            args.default_label,
            args.save_annotated_video,
            args.export_per_frame_csv,
        )

        seg_default, sum_default = run_video_for_model(
            video_path=args.video_path,
            model_path=args.default_model,
            model_label=args.default_label,
            contour=contour,
            output_video_path=default_video_out,
            per_frame_csv_path=default_per_frame,
            segments_csv_path=default_segments,
            summary_json_path=default_summary,
            fps=fps,
            tracker_cfg=args.tracker,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            out_dir=args.out_dir,
            save_annotated_video=args.save_annotated_video,
            export_per_frame_csv=args.export_per_frame_csv,
            vehicle_class_filter=args.default_class_filter,
        )
        ran["default"] = (seg_default, sum_default)

    if not args.skip_plots and "tuned" in ran and "default" in ran:
        plot_comparison(
            ran["tuned"][0],
            ran["default"][0],
            label_a=args.tuned_label,
            label_b=args.default_label,
            out_prefix=os.path.join(args.out_dir, "compare"),
        )

    for key, (_, summary) in ran.items():
        print(f"\n=== Summary ({key}) ===")
        print(json.dumps(summary, indent=2))

    print(f"\nSaved outputs under: {args.out_dir}")


if __name__ == "__main__":
    main()