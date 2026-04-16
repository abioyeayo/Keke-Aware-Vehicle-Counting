from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Project-aware defaults
# -----------------------------------------------------------------------------
DOMINANT_CLASSES: Tuple[str, ...] = ("bus", "car", "keke", "truck")
DEFAULT_COCO_VEHICLE_CLASS_IDS: List[int] = [2, 3, 5, 7]

VIDEO_REGISTRY: Dict[str, dict] = {
    "yola_road": {
        "video_path": "./datasets/test/20250627_121728h_yola_road.mp4",
        "baseline_path": "./baseline/yola_road_mp4_baseline_vehicle_count.csv",
        "region_points": [(20, 400), (1080, 400), (1080, 360), (20, 360)],
        "scene": "daytime, moderate traffic, high visibility",
    },
    "mubi_road": {
        "video_path": "./datasets/test/20260131_170349h_mubi_road.mp4",
        "baseline_path": "./baseline/mubi_road_mp4_baseline_vehicle_count.csv",
        "region_points": [(320, 600), (1280, 600), (1280, 560), (320, 560)],
        "scene": "evening, higher congestion, cloudy / low visibility",
    },
}

MODEL_REGISTRY: Dict[str, dict] = {
    "yolo11l_default": {
        "display_name": "YOLO11l default",
        "architecture": "YOLO11l",
        "variant": "default",
        "model_path": "yolo11l.pt",
        "class_filter": DEFAULT_COCO_VEHICLE_CLASS_IDS,
    },
    "yolo11l_tuned": {
        "display_name": "YOLO11l tuned",
        "architecture": "YOLO11l",
        "variant": "tuned",
        "model_path": "runs/detect/train_yolo11l/weights/best.pt",
        "class_filter": None,
    },
    "yolo26l_default": {
        "display_name": "YOLO26l default",
        "architecture": "YOLO26l",
        "variant": "default",
        "model_path": "yolo26l.pt",
        "class_filter": DEFAULT_COCO_VEHICLE_CLASS_IDS,
    },
    "yolo26l_tuned": {
        "display_name": "YOLO26l tuned",
        "architecture": "YOLO26l",
        "variant": "tuned",
        "model_path": "runs/detect/train_yolo26l/weights/best.pt",
        "class_filter": None,
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def region_contour(points: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def xyxy_centroid(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def in_region(contour: np.ndarray, cx: float, cy: float) -> bool:
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def parse_region_points(text: str) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for token in text.strip().split():
        x_str, y_str = token.split(",")
        points.append((int(x_str), int(y_str)))
    if len(points) < 3:
        raise argparse.ArgumentTypeError("Region must contain at least 3 points.")
    return points


def slugify(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    while "__" in "".join(cleaned):
        cleaned = list("".join(cleaned).replace("__", "_"))
    return "".join(cleaned).strip("_")


@dataclass
class TrackState:
    cls_name: str
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

    def start_segment(self, t: float, frame_idx: int, cx: float, cy: float) -> None:
        self.inside = True
        self.enter_time = t
        self.enter_frame = frame_idx
        self.enter_cx = cx
        self.enter_cy = cy
        self.last_cx = cx
        self.last_cy = cy
        self.conf_sum_inside = 0.0
        self.conf_max_inside = 0.0
        self.frames_inside = 0

    def update_inside(self, conf: float, cx: float, cy: float) -> None:
        self.frames_inside += 1
        self.conf_sum_inside += conf
        self.conf_max_inside = max(self.conf_max_inside, conf)
        self.last_cx = cx
        self.last_cy = cy

    def end_segment(self, t_exit: float, frame_exit: int, track_id: int) -> None:
        if self.enter_time is None or self.enter_frame is None:
            return
        duration = t_exit - self.enter_time
        avg_conf = self.conf_sum_inside / self.frames_inside if self.frames_inside > 0 else 0.0
        self.segments.append(
            {
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
            }
        )
        self.inside = False
        self.enter_time = None
        self.enter_frame = None
        self.enter_cx = None
        self.enter_cy = None
        self.last_cx = None
        self.last_cy = None
        self.conf_sum_inside = 0.0
        self.conf_max_inside = 0.0
        self.frames_inside = 0


@dataclass
class Experiment:
    video_key: str
    video_path: str
    baseline_path: str
    scene: str
    model_key: str
    model_path: str
    model_display_name: str
    architecture: str
    variant: str
    tracker: str
    region_mode: str
    region_points: List[Tuple[int, int]]
    class_filter: Optional[List[int]]
    detector_conf: float
    detector_iou: float
    save_video: bool
    export_per_frame: bool
    max_frames: Optional[int] = None

    @property
    def tracker_name(self) -> str:
        return Path(self.tracker).stem

    @property
    def experiment_key(self) -> str:
        return "__".join(
            [
                slugify(self.video_key),
                slugify(self.model_key),
                slugify(self.tracker_name),
                slugify(self.region_mode),
            ]
        )


# -----------------------------------------------------------------------------
# Experiment matrix builders
# -----------------------------------------------------------------------------

def build_paper_core_matrix(
    videos: Sequence[str],
    detector_conf: float,
    detector_iou: float,
    save_video: bool,
    export_per_frame: bool,
    max_frames: Optional[int],
) -> List[Experiment]:
    """
    A focused, reviewer-driven set of experiments with manageable runtime:
      - 4 core detector comparisons with BoT-SORT + ROI.
      - 2 tracker comparisons (ByteTrack) on tuned models.
      - 2 ROI ablations (full-frame counting region) on tuned models.

    This gives direct evidence for:
      1) added keke class value,
      2) tracker choice,
      3) ROI contribution.
    """
    rows = [
        ("yolo11l_default", "botsort.yaml", "roi"),
        ("yolo11l_tuned", "botsort.yaml", "roi"),
        ("yolo26l_default", "botsort.yaml", "roi"),
        ("yolo26l_tuned", "botsort.yaml", "roi"),
        ("yolo11l_tuned", "bytetrack.yaml", "roi"),
        ("yolo26l_tuned", "bytetrack.yaml", "roi"),
        ("yolo11l_tuned", "botsort.yaml", "full_frame"),
        ("yolo26l_tuned", "botsort.yaml", "full_frame"),
    ]
    return make_experiments(
        videos=videos,
        rows=rows,
        detector_conf=detector_conf,
        detector_iou=detector_iou,
        save_video=save_video,
        export_per_frame=export_per_frame,
        max_frames=max_frames,
    )



def build_full_matrix(
    videos: Sequence[str],
    detector_conf: float,
    detector_iou: float,
    save_video: bool,
    export_per_frame: bool,
    max_frames: Optional[int],
) -> List[Experiment]:
    rows = []
    for model_key in MODEL_REGISTRY:
        for tracker in ("botsort.yaml", "bytetrack.yaml"):
            for region_mode in ("roi", "full_frame"):
                rows.append((model_key, tracker, region_mode))
    return make_experiments(
        videos=videos,
        rows=rows,
        detector_conf=detector_conf,
        detector_iou=detector_iou,
        save_video=save_video,
        export_per_frame=export_per_frame,
        max_frames=max_frames,
    )



def make_experiments(
    videos: Sequence[str],
    rows: Sequence[Tuple[str, str, str]],
    detector_conf: float,
    detector_iou: float,
    save_video: bool,
    export_per_frame: bool,
    max_frames: Optional[int],
) -> List[Experiment]:
    experiments: List[Experiment] = []
    for video_key in videos:
        video_cfg = VIDEO_REGISTRY[video_key]
        for model_key, tracker, region_mode in rows:
            model_cfg = MODEL_REGISTRY[model_key]
            experiments.append(
                Experiment(
                    video_key=video_key,
                    video_path=video_cfg["video_path"],
                    baseline_path=video_cfg["baseline_path"],
                    scene=video_cfg["scene"],
                    model_key=model_key,
                    model_path=model_cfg["model_path"],
                    model_display_name=model_cfg["display_name"],
                    architecture=model_cfg["architecture"],
                    variant=model_cfg["variant"],
                    tracker=tracker,
                    region_mode=region_mode,
                    region_points=list(video_cfg["region_points"]),
                    class_filter=model_cfg["class_filter"],
                    detector_conf=detector_conf,
                    detector_iou=detector_iou,
                    save_video=save_video,
                    export_per_frame=export_per_frame,
                    max_frames=max_frames,
                )
            )
    return experiments


# -----------------------------------------------------------------------------
# Core video runner
# -----------------------------------------------------------------------------

def resolve_contour(region_mode: str, frame_width: int, frame_height: int, roi_points: Sequence[Tuple[int, int]]) -> np.ndarray:
    if region_mode == "roi":
        points = roi_points
    elif region_mode == "full_frame":
        points = [(0, 0), (frame_width - 1, 0), (frame_width - 1, frame_height - 1), (0, frame_height - 1)]
    else:
        raise ValueError(f"Unsupported region_mode: {region_mode}")
    return region_contour(points)



def run_experiment(experiment: Experiment, output_root: Path) -> dict:
    exp_dir = output_root / experiment.video_key / experiment.experiment_key
    ensure_dir(exp_dir)

    meta_path = exp_dir / "experiment_meta.json"
    segments_path = exp_dir / "region_segments.csv"
    per_frame_path = exp_dir / "per_frame.csv"
    summary_path = exp_dir / "summary.json"
    annotated_path = exp_dir / "annotated.mp4"

    cap = cv2.VideoCapture(experiment.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {experiment.video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    contour = resolve_contour(experiment.region_mode, frame_width, frame_height, experiment.region_points)

    model = YOLO(experiment.model_path)
    names = model.names

    writer = None
    if experiment.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (frame_width, frame_height))

    track_states: Dict[int, TrackState] = {}
    per_frame_rows: List[dict] = []
    all_segments: List[dict] = []

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if experiment.max_frames is not None and frame_idx >= experiment.max_frames:
            break

        timestamp_s = frame_idx / fps if fps > 0 else float(frame_idx)
        results = model.track(
            source=frame,
            persist=True,
            tracker=experiment.tracker,
            conf=experiment.detector_conf,
            iou=experiment.detector_iou,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        annotated = result.plot()
        cv2.polylines(annotated, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            tids = boxes.id.cpu().numpy().astype(int)

            for bb, conf, cls_id, tid in zip(xyxy, confs, clss, tids):
                if experiment.class_filter is not None and cls_id not in experiment.class_filter:
                    continue

                cls_name = names.get(cls_id, str(cls_id))
                cx, cy = xyxy_centroid(bb)
                inside = in_region(contour, cx, cy)

                if tid not in track_states:
                    track_states[tid] = TrackState(cls_name=cls_name)

                state = track_states[tid]
                if inside and not state.inside:
                    state.start_segment(timestamp_s, frame_idx, cx, cy)
                if inside and state.inside:
                    state.update_inside(float(conf), cx, cy)
                if (not inside) and state.inside:
                    state.end_segment(timestamp_s, frame_idx, tid)

                if experiment.export_per_frame:
                    x1, y1, x2, y2 = bb
                    per_frame_rows.append(
                        {
                            "timestamp_s": timestamp_s,
                            "frame": frame_idx,
                            "track_id": int(tid),
                            "class": cls_name,
                            "conf": float(conf),
                            "in_region": bool(inside),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "cx": float(cx),
                            "cy": float(cy),
                        }
                    )

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1

    end_time_s = (frame_idx - 1) / fps if frame_idx > 0 and fps > 0 else 0.0
    for tid, state in track_states.items():
        if state.inside:
            state.end_segment(end_time_s, frame_idx - 1, tid)
        all_segments.extend(state.segments)

    elapsed = max(time.time() - t0, 1e-9)
    processing_fps = frame_idx / elapsed if frame_idx > 0 else 0.0

    cap.release()
    if writer is not None:
        writer.release()

    if experiment.export_per_frame:
        pd.DataFrame(per_frame_rows).to_csv(per_frame_path, index=False)

    seg_df = pd.DataFrame(all_segments)
    if seg_df.empty:
        seg_df = pd.DataFrame(
            columns=[
                "track_id",
                "class",
                "enter_time_s",
                "exit_time_s",
                "duration_s",
                "enter_frame",
                "exit_frame",
                "enter_cx",
                "enter_cy",
                "exit_cx",
                "exit_cy",
                "avg_conf_in_region",
                "max_conf_in_region",
                "frames_in_region",
            ]
        )
    seg_df.insert(0, "experiment_key", experiment.experiment_key)
    seg_df.insert(1, "video_key", experiment.video_key)
    seg_df.insert(2, "model_key", experiment.model_key)
    seg_df.insert(3, "model_display_name", experiment.model_display_name)
    seg_df.insert(4, "tracker", experiment.tracker_name)
    seg_df.insert(5, "region_mode", experiment.region_mode)
    seg_df.to_csv(segments_path, index=False)

    summary = {
        "experiment_key": experiment.experiment_key,
        "video_key": experiment.video_key,
        "video_path": experiment.video_path,
        "baseline_path": experiment.baseline_path,
        "scene": experiment.scene,
        "model_key": experiment.model_key,
        "model_display_name": experiment.model_display_name,
        "architecture": experiment.architecture,
        "variant": experiment.variant,
        "model_path": experiment.model_path,
        "tracker": experiment.tracker_name,
        "tracker_cfg": experiment.tracker,
        "region_mode": experiment.region_mode,
        "region_points": experiment.region_points,
        "detector_conf": experiment.detector_conf,
        "detector_iou": experiment.detector_iou,
        "class_filter": experiment.class_filter,
        "frames_processed": frame_idx,
        "video_total_frames": total_frames,
        "video_fps": fps,
        "processing_seconds": elapsed,
        "processing_fps": processing_fps,
        "segments": int(len(seg_df)),
        "unique_tracks_with_region_segments": int(seg_df["track_id"].nunique()) if not seg_df.empty else 0,
        "total_dwell_time_s": float(seg_df["duration_s"].sum()) if not seg_df.empty else 0.0,
        "avg_dwell_time_s": float(seg_df["duration_s"].mean()) if not seg_df.empty else 0.0,
        "avg_conf_in_region": float(seg_df["avg_conf_in_region"].mean()) if not seg_df.empty else 0.0,
        "outputs": {
            "segments_csv": str(segments_path),
            "summary_json": str(summary_path),
            "per_frame_csv": str(per_frame_path) if experiment.export_per_frame else None,
            "annotated_video": str(annotated_path) if experiment.save_video else None,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(experiment), f, indent=2)
    return summary


# -----------------------------------------------------------------------------
# Legacy import mode
# -----------------------------------------------------------------------------

def infer_video_key_from_dir(name: str) -> Optional[str]:
    lowered = name.lower()
    if "yola" in lowered:
        return "yola_road"
    if "mubi" in lowered:
        return "mubi_road"
    return None



def infer_model_key_from_dir(name: str, label: str) -> Optional[str]:
    lowered = name.lower()
    if "yolo11l" in lowered:
        return "yolo11l_tuned" if label == "tuned" else "yolo11l_default"
    if "yolo26l" in lowered or "keke_rev" in lowered:
        return "yolo26l_tuned" if label == "tuned" else "yolo26l_default"
    return None



def import_existing_results(existing_dirs: Sequence[str], output_root: Path) -> pd.DataFrame:
    records: List[dict] = []
    for root in existing_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARN] Skipping missing directory: {root_path}")
            continue

        video_key = infer_video_key_from_dir(root_path.name)
        if video_key is None:
            print(f"[WARN] Could not infer video from directory name: {root_path.name}")
            continue

        for label in ("default", "tuned"):
            seg_src = root_path / f"{label}_region_segments.csv"
            sum_src = root_path / f"{label}_summary.json"
            pf_src = root_path / f"{label}_per_frame.csv"
            vid_src = root_path / f"annotated_{label}.mp4"
            if not seg_src.exists() or not sum_src.exists():
                continue

            model_key = infer_model_key_from_dir(root_path.name, label)
            if model_key is None:
                print(f"[WARN] Could not infer model from directory name: {root_path.name}")
                continue

            model_cfg = MODEL_REGISTRY[model_key]
            video_cfg = VIDEO_REGISTRY[video_key]
            tracker_name = "botsort"
            region_mode = "roi"
            exp = Experiment(
                video_key=video_key,
                video_path=video_cfg["video_path"],
                baseline_path=video_cfg["baseline_path"],
                scene=video_cfg["scene"],
                model_key=model_key,
                model_path=model_cfg["model_path"],
                model_display_name=model_cfg["display_name"],
                architecture=model_cfg["architecture"],
                variant=model_cfg["variant"],
                tracker="botsort.yaml",
                region_mode=region_mode,
                region_points=list(video_cfg["region_points"]),
                class_filter=model_cfg["class_filter"],
                detector_conf=0.25,
                detector_iou=0.45,
                save_video=vid_src.exists(),
                export_per_frame=pf_src.exists(),
                max_frames=None,
            )
            exp_dir = output_root / exp.video_key / exp.experiment_key
            ensure_dir(exp_dir)

            shutil.copy2(seg_src, exp_dir / "region_segments.csv")
            shutil.copy2(sum_src, exp_dir / "summary.json")
            if pf_src.exists():
                shutil.copy2(pf_src, exp_dir / "per_frame.csv")
            if vid_src.exists():
                shutil.copy2(vid_src, exp_dir / "annotated.mp4")
            with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
                json.dump(asdict(exp), f, indent=2)

            summary = json.loads(sum_src.read_text(encoding="utf-8"))
            summary.update(
                {
                    "experiment_key": exp.experiment_key,
                    "video_key": exp.video_key,
                    "baseline_path": exp.baseline_path,
                    "scene": exp.scene,
                    "model_key": exp.model_key,
                    "model_display_name": exp.model_display_name,
                    "architecture": exp.architecture,
                    "variant": exp.variant,
                    "tracker": tracker_name,
                    "region_mode": region_mode,
                    "detector_conf": summary.get("conf_thres", 0.25),
                    "detector_iou": summary.get("iou_thres", 0.45),
                    "outputs": {
                        "segments_csv": str(exp_dir / "region_segments.csv"),
                        "summary_json": str(exp_dir / "summary.json"),
                        "per_frame_csv": str(exp_dir / "per_frame.csv") if pf_src.exists() else None,
                        "annotated_video": str(exp_dir / "annotated.mp4") if vid_src.exists() else None,
                    },
                }
            )
            (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            records.append(summary)
            print(f"[IMPORTED] {root_path.name} -> {exp.experiment_key}")

    index_df = pd.DataFrame(records)
    if not index_df.empty:
        index_df.sort_values(["video_key", "architecture", "variant"], inplace=True)
        index_df.to_csv(output_root / "experiment_index.csv", index=False)
        manifest = {
            "mode": "import_existing",
            "experiments": index_df["experiment_key"].tolist(),
            "videos": sorted(index_df["video_key"].unique().tolist()),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return index_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reviewer-oriented counting ablation suite for all configured YOLO models and videos."
    )
    parser.add_argument(
        "--output-root",
        default="./results/ablation_suite",
        help="Directory where experiment folders and suite metadata will be written.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper_core", "full_matrix"],
        default="paper_core",
        help="paper_core is the recommended high-reward subset; full_matrix is exhaustive.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["yola_road", "mubi_road"],
        choices=sorted(VIDEO_REGISTRY.keys()),
        help="Videos to include in the suite.",
    )
    parser.add_argument("--detector-conf", type=float, default=0.25, help="YOLO tracking confidence threshold.")
    parser.add_argument("--detector-iou", type=float, default=0.45, help="YOLO tracking IoU threshold.")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated videos. Off by default because it increases runtime and storage.",
    )
    parser.add_argument(
        "--export-per-frame",
        action="store_true",
        help="Export per-frame CSVs. Off by default because they are large.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug cap on the number of frames processed per video.",
    )
    parser.add_argument(
        "--import-existing",
        nargs="*",
        default=None,
        help=(
            "Instead of rerunning inference, import existing result directories "
            "such as ./results/yolo11l_yola_road ./results/yolo26l_yola_road ./results/yolo11l_mubi_road ./results/yolo26l_mubi_road"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if args.import_existing is not None and len(args.import_existing) > 0:
        index_df = import_existing_results(args.import_existing, output_root)
        if index_df.empty:
            raise RuntimeError("No existing result directories were successfully imported.")
        print(f"\nImported {len(index_df)} experiments into: {output_root}")
        return

    if args.preset == "paper_core":
        experiments = build_paper_core_matrix(
            videos=args.videos,
            detector_conf=args.detector_conf,
            detector_iou=args.detector_iou,
            save_video=args.save_video,
            export_per_frame=args.export_per_frame,
            max_frames=args.max_frames,
        )
    else:
        experiments = build_full_matrix(
            videos=args.videos,
            detector_conf=args.detector_conf,
            detector_iou=args.detector_iou,
            save_video=args.save_video,
            export_per_frame=args.export_per_frame,
            max_frames=args.max_frames,
        )

    records: List[dict] = []
    for idx, experiment in enumerate(experiments, start=1):
        print(f"[{idx}/{len(experiments)}] {experiment.experiment_key}")
        summary = run_experiment(experiment, output_root)
        records.append(summary)
        print(
            f"    segments={summary['segments']} | unique_tracks={summary['unique_tracks_with_region_segments']} "
            f"| proc_fps={summary['processing_fps']:.2f}"
        )

    index_df = pd.DataFrame(records)
    index_df.sort_values(["video_key", "architecture", "variant", "tracker", "region_mode"], inplace=True)
    index_df.to_csv(output_root / "experiment_index.csv", index=False)

    manifest = {
        "preset": args.preset,
        "videos": args.videos,
        "detector_conf": args.detector_conf,
        "detector_iou": args.detector_iou,
        "save_video": args.save_video,
        "export_per_frame": args.export_per_frame,
        "max_frames": args.max_frames,
        "num_experiments": len(experiments),
        "experiments": [experiment.experiment_key for experiment in experiments],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSuite complete. Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
