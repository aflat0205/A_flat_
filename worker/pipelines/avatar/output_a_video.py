"""Output A pipeline: face-scan video -> stylized character video.

Idempotent: re-running with the same job_dir skips completed stages
(tracked via manifest.json).
"""
import json
import time
from pathlib import Path

from pipelines.avatar.style_config import get_style
from pipelines.avatar.stages import (
    decode,
    face_landmarks,
    depth_estimation,
    stylize,
    postprocess,
    encode,
)
import config


def _stage_done(manifest_path: Path, name: str) -> bool:
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        return m.get(name, {}).get("done", False)
    return False


def _mark_done(manifest_path: Path, name: str, meta: dict = None):
    m = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
    m[name] = {"done": True, "timestamp": time.time(), **(meta or {})}
    manifest_path.write_text(json.dumps(m, indent=2))


def run(
    input_video: Path,
    output_video: Path,
    style_id: str,
    job_dir: Path,
    seed: int = 42,
) -> Path:
    """Run the complete Output A pipeline.

    Args:
        input_video: Path to input face-scan MP4.
        output_video: Path for final output MP4.
        style_id: One of "beauty-realistic", "promptable-avatar", "animated-anime".
        job_dir: Working directory for intermediate files.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output MP4 file.
    """
    style = get_style(style_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = job_dir / "manifest.json"

    # ── Stage 1: Decode ──
    frames_dir = job_dir / "frames"
    if not _stage_done(manifest, "decode"):
        print("[1/6] Extracting frames...")
        frame_paths, video_meta = decode.extract_frames(input_video, frames_dir)
        _mark_done(manifest, "decode", {
            "fps": video_meta.fps,
            "width": video_meta.width,
            "height": video_meta.height,
            "frame_count": video_meta.frame_count,
            "duration": video_meta.duration,
        })
    else:
        print("[1/6] Decode: cached")
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        m = json.loads(manifest.read_text())["decode"]
        video_meta = decode.VideoMeta(
            width=m["width"],
            height=m["height"],
            fps=m["fps"],
            duration=m["duration"],
            frame_count=m["frame_count"],
        )

    # ── Stage 2: Face Landmarks ──
    pose_dir = job_dir / "pose"
    if not _stage_done(manifest, "face_landmarks"):
        print("[2/6] Detecting face landmarks...")
        pose_paths = face_landmarks.process_frames(
            model_path=config.FACE_LANDMARKER_PATH,
            frame_paths=frame_paths,
            output_dir=pose_dir,
            width=video_meta.width,
            height=video_meta.height,
        )
        _mark_done(manifest, "face_landmarks", {"count": len(pose_paths)})
    else:
        print("[2/6] Face landmarks: cached")
        pose_paths = sorted(pose_dir.glob("pose_*.png"))

    # ── Stage 3: Depth Estimation ──
    depth_dir = job_dir / "depth"
    if not _stage_done(manifest, "depth_estimation"):
        print("[3/6] Estimating depth maps...")
        depth_paths = depth_estimation.process_frames(
            model_id=config.DEPTH_MODEL_ID,
            device=config.DEVICE,
            frame_paths=frame_paths,
            output_dir=depth_dir,
        )
        _mark_done(manifest, "depth_estimation", {"count": len(depth_paths)})
    else:
        print("[3/6] Depth estimation: cached")
        depth_paths = sorted(depth_dir.glob("depth_*.png"))

    # ── Stage 4: Stylize ──
    styled_dir = job_dir / "styled"
    if not _stage_done(manifest, "stylize"):
        print(f"[4/6] Stylizing frames with '{style.display_name}'...")
        styled_paths = stylize.process_frames(
            style=style,
            device=config.DEVICE,
            dtype=config.DTYPE,
            frame_paths=frame_paths,
            openpose_paths=pose_paths,
            depth_paths=depth_paths,
            output_dir=styled_dir,
            seed=seed,
        )
        _mark_done(manifest, "stylize", {
            "count": len(styled_paths),
            "style_id": style_id,
        })
    else:
        print("[4/6] Stylize: cached")
        styled_paths = sorted(styled_dir.glob("styled_*.png"))

    # ── Stage 5: Post-process ──
    final_dir = job_dir / "final"
    if not _stage_done(manifest, "postprocess"):
        print("[5/6] Post-processing (color match + temporal smooth)...")
        postprocess.process_frames(
            styled_paths=styled_paths,
            original_paths=frame_paths,
            output_dir=final_dir,
            color_match_strength=style.color_match_strength,
            temporal_blend_frames=style.temporal_blend_frames,
        )
        _mark_done(manifest, "postprocess")
    else:
        print("[5/6] Post-process: cached")

    # ── Stage 6: Encode ──
    if not _stage_done(manifest, "encode"):
        print("[6/6] Encoding output video...")
        encode.encode_video(
            frame_dir=final_dir,
            output_path=output_video,
            fps=video_meta.fps,
        )
        _mark_done(manifest, "encode", {"output": str(output_video)})
    else:
        print("[6/6] Encode: cached")

    print(f"\nDone! Output: {output_video}")
    return output_video
