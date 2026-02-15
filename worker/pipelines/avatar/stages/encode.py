import subprocess
from pathlib import Path


def encode_video(
    frame_dir: Path,
    output_path: Path,
    fps: float,
) -> Path:
    """Encode PNG frames into an H.264 MP4 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(frame_dir.glob("final_*.png"))
    if not frames:
        raise FileNotFoundError(f"No final_*.png frames found in {frame_dir}")

    # Use ffmpeg concat demuxer for reliable frame ordering
    filelist = frame_dir / "_filelist.txt"
    with open(filelist, "w") as f:
        for fp in frames:
            f.write(f"file '{fp.name}'\n")
            f.write(f"duration {1.0 / fps}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(filelist),
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    filelist.unlink(missing_ok=True)
    return output_path
