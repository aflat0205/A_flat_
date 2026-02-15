import json
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


def probe_video(video_path: Path) -> VideoMeta:
    """Use ffprobe to get video metadata (fps, resolution, duration)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = map(int, vstream["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(info["format"]["duration"])
    return VideoMeta(
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps=fps,
        duration=duration,
        frame_count=int(fps * duration),
    )


def extract_frames(video_path: Path, output_dir: Path) -> tuple[list[Path], VideoMeta]:
    """Extract all frames as PNG files. Returns (sorted_frame_paths, metadata)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video_path)
    pattern = str(output_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={meta.fps}",
        "-vsync", "vfr",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    frames = sorted(output_dir.glob("frame_*.png"))
    meta.frame_count = len(frames)
    return frames, meta
