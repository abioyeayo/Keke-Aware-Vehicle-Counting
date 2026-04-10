import os
import cv2

def extract_one_frame_per_second(video_path: str, out_dir: str, prefix: str = "frame"):
    os.makedirs(out_dir, exist_ok=True)

    target_fps = 5

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        raise RuntimeError("Could not read FPS from video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if total_frames > 0 else None

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    if duration_s is not None:
        print(f"Approx duration: {duration_s:.2f}s  (~{int(duration_s * target_fps)} images at {target_fps} fps)")
    
    saved = 0
    sample_idx = 0

    while True:
        # # Seek to the frame at time = sec seconds
        # frame_idx = int(round(sec * fps))
        frame_idx = int(round(sample_idx * fps/target_fps))
        if total_frames and frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # out_path = os.path.join(out_dir, f"{prefix}_{sample_idx:06d}.jpg")
        t = sample_idx / target_fps  # time in seconds
        out_path = os.path.join(out_dir, f"{prefix}_{t:07.1f}s.jpg")
        ok = cv2.imwrite(out_path, frame)
        if not ok:
            raise RuntimeError(f"Failed to write image: {out_path}")

        saved += 1
        sample_idx += 1

    cap.release()
    print(f"Saved {saved} images to: {out_dir}")

if __name__ == "__main__":

    # video = "./exports6_yola_shopping_complex/annotated_tuned.mp4"
    # out_dir = "./baseline/6_yola_shopping_complex_mp4_image_frames"

    video = "./results/yolo26l_mubi_road/annotated_tuned.mp4"
    out_dir = "./baseline/mubi_road_mp4_image_frames"

    extract_one_frame_per_second(video, out_dir)
