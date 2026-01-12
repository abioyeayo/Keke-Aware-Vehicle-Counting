import cv2
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# -------------------------
# Configuration
# -------------------------
VIDEO_PATH = "./datasets/test/yola_road.mp4"

# Rectangle region (same points you used)
REGION_POINTS = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Compare these two models:
TUNED_MODEL = "runs/detect/train/weights/best.pt"
DEFAULT_MODEL = "yolo11l.pt"  # change to yolo11n.pt/yolov8n.pt if you prefer

# Vehicle classes for COCO (if using a COCO model)
# COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Tracking config
TRACKER_CFG = "botsort.yaml"   # or "bytetrack.yaml"
CONF_THRES = 0.25
IOU_THRES = 0.45

# Outputs
OUT_DIR = "./results"
SAVE_ANNOTATED_VIDEO = True
EXPORT_PER_FRAME_CSV = True  # can be large for long videos


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)

def region_contour(points: List[Tuple[int, int]]) -> np.ndarray:
    # cv2.pointPolygonTest expects shape (N,1,2)
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

def in_region(contour: np.ndarray, cx: float, cy: float) -> bool:
    # returns +1, 0, -1 (inside, on edge, outside)
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0

def xyxy_centroid(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class TrackState:
    cls_name: str
    ever_entered: bool = False
    inside: bool = False
    enter_time: Optional[float] = None
    enter_frame: Optional[int] = None

    # NEW:
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

        # NEW:
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

        # NEW:
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

            # NEW:
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
    model_path: str,
    model_label: str,
    contour: np.ndarray,
    output_video_path: Optional[str],
    per_frame_csv_path: str,
    segments_csv_path: str,
    summary_json_path: str,
    fps: float,
    vehicle_class_filter: Optional[List[int]] = None,
):
    ensure_dir(OUT_DIR)

    model = YOLO(model_path)
    names = model.names  # dict: id -> name

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Error reading video file: {VIDEO_PATH}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path and SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    track_states: Dict[int, TrackState] = {}
    per_frame_rows = []
    all_segments = []

    frame_idx = 0
    t0 = time.time()

    # Main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps

        # Track on this frame
        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False,
        )
        r0 = results[0]

        # If no detections, still need to potentially end segments? (We keep segments open until object leaves region)
        # However, with tracking, absence means track is lost. If you want to close on loss, you can do that here.
        boxes = r0.boxes

        # Draw region on the frame
        annotated = r0.plot()
        cv2.polylines(annotated, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            tids = boxes.id.cpu().numpy().astype(int)

            for bb, conf, cls_id, tid in zip(xyxy, confs, clss, tids):
                # Optional class filtering (useful for COCO default model)
                if vehicle_class_filter is not None and cls_id not in vehicle_class_filter:
                    continue

                cls_name = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = bb
                cx, cy = xyxy_centroid(bb)
                inside = in_region(contour, cx, cy)

                # init track state if new
                if tid not in track_states:
                    track_states[tid] = TrackState(cls_name=cls_name)
                else:
                    # if class name changes, keep first (or update—your call)
                    pass

                st = track_states[tid]

                # segment logic
                if inside and not st.inside:
                    st.start_segment(timestamp_s, frame_idx, cx, cy)

                if inside and st.inside:
                    st.update_inside(float(conf), cx, cy)

                if (not inside) and st.inside:
                    st.end_segment(timestamp_s, frame_idx, tid)

                # per-frame export
                if EXPORT_PER_FRAME_CSV:
                    per_frame_rows.append({
                        "model": model_label,
                        "timestamp_s": timestamp_s,
                        "frame": frame_idx,
                        "track_id": tid,
                        "class": cls_name,
                        "conf": float(conf),
                        "in_region": bool(inside),

                        # NEW:
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "cx": float(cx), "cy": float(cy),
                    })

        # write video
        if writer is not None:
            writer.write(annotated)

        frame_idx += 1

    # Close any segments still open at end of video
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

    # Save CSVs
    if EXPORT_PER_FRAME_CSV:
        pd.DataFrame(per_frame_rows).to_csv(per_frame_csv_path, index=False)

    seg_df = pd.DataFrame(all_segments)
    seg_df.insert(0, "model", model_label)
    seg_df.to_csv(segments_csv_path, index=False)

    # Summary
    total_segments = len(seg_df)
    total_unique_tracks = seg_df["track_id"].nunique() if total_segments else 0
    total_dwell = float(seg_df["duration_s"].sum()) if total_segments else 0.0
    avg_dwell = float(seg_df["duration_s"].mean()) if total_segments else 0.0
    avg_conf = float(seg_df["avg_conf_in_region"].mean()) if total_segments else 0.0

    summary = {
        "model": model_label,
        "model_path": model_path,
        "frames_processed": frame_idx,
        "video_fps": fps,
        "processing_seconds": elapsed,
        "processing_fps": processing_fps,
        "segments": total_segments,
        "unique_tracks_with_region_segments": total_unique_tracks,
        "total_dwell_time_s": total_dwell,
        "avg_dwell_time_s": avg_dwell,
        "avg_conf_in_region": avg_conf,
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return seg_df, summary


def plot_comparison(seg_a: pd.DataFrame, seg_b: pd.DataFrame, label_a: str, label_b: str, out_prefix: str):
    # 1) Counts over time (bin by second using enter_time)
    def counts_over_time(seg: pd.DataFrame):
        if seg.empty:
            return pd.DataFrame(columns=["t", "count"])
        t = np.floor(seg["enter_time_s"].values).astype(int)
        s = pd.Series(1, index=t).groupby(level=0).sum().sort_index()
        return pd.DataFrame({"t": s.index.values, "count": s.values})

    cA = counts_over_time(seg_a)
    cB = counts_over_time(seg_b)

    plt.figure()
    if not cA.empty:
        plt.plot(cA["t"], cA["count"], label=label_a)
    if not cB.empty:
        plt.plot(cB["t"], cB["count"], label=label_b)
    plt.xlabel("Time (s, binned)")
    plt.ylabel("Region entries (segments started)")
    plt.title("Region entry counts over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_counts_over_time.png", dpi=200)
    plt.close()

    # 2) Dwell time distribution (hist)
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

    # 3) Confidence distribution (avg_conf_in_region)
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


def main():
    ensure_dir(OUT_DIR)

    # Read fps from video
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Error reading video file"
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    contour = region_contour(REGION_POINTS)

    # Decide whether to filter classes.
    # If your tuned model is trained ONLY for vehicles, set vehicle_class_filter=None for tuned.
    # For the COCO default model, filtering helps keep only vehicles.
    tuned_filter = None
    default_filter = COCO_VEHICLE_CLASS_IDS

    tuned_label = "tuned"
    default_label = "default"

    tuned_video_out = f"{OUT_DIR}/annotated_{tuned_label}.mp4"
    default_video_out = f"{OUT_DIR}/annotated_{default_label}.mp4"

    tuned_per_frame = f"{OUT_DIR}/{tuned_label}_per_frame.csv"
    default_per_frame = f"{OUT_DIR}/{default_label}_per_frame.csv"

    tuned_segments = f"{OUT_DIR}/{tuned_label}_region_segments.csv"
    default_segments = f"{OUT_DIR}/{default_label}_region_segments.csv"

    tuned_summary = f"{OUT_DIR}/{tuned_label}_summary.json"
    default_summary = f"{OUT_DIR}/{default_label}_summary.json"

    seg_tuned, sum_tuned = run_video_for_model(
        model_path=TUNED_MODEL,
        model_label=tuned_label,
        contour=contour,
        output_video_path=tuned_video_out,
        per_frame_csv_path=tuned_per_frame,
        segments_csv_path=tuned_segments,
        summary_json_path=tuned_summary,
        fps=fps,
        vehicle_class_filter=tuned_filter,
    )

    seg_default, sum_default = run_video_for_model(
        model_path=DEFAULT_MODEL,
        model_label=default_label,
        contour=contour,
        output_video_path=default_video_out,
        per_frame_csv_path=default_per_frame,
        segments_csv_path=default_segments,
        summary_json_path=default_summary,
        fps=fps,
        vehicle_class_filter=default_filter,
    )

    # Comparison plots
    plot_comparison(
        seg_tuned, seg_default,
        label_a="Tuned", label_b="Default",
        out_prefix=f"{OUT_DIR}/compare"
    )

    # Print a quick console summary
    print("\n=== Summary (Tuned) ===")
    print(json.dumps(sum_tuned, indent=2))
    print("\n=== Summary (Default) ===")
    print(json.dumps(sum_default, indent=2))
    print(f"\nSaved CSV/plots/videos under: {OUT_DIR}")


if __name__ == "__main__":
    main()
