import cv2
import numpy as np
from pathlib import Path


def color_transfer(
    source: np.ndarray, target: np.ndarray, strength: float
) -> np.ndarray:
    """Transfer color statistics from target (original) to source (styled).
    Uses LAB color space mean/std transfer, blended by strength."""
    if strength <= 0:
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean = src_lab[:, :, ch].mean()
        src_std = src_lab[:, :, ch].std() + 1e-6
        tgt_mean = tgt_lab[:, :, ch].mean()
        tgt_std = tgt_lab[:, :, ch].std() + 1e-6
        transferred = (src_lab[:, :, ch] - src_mean) * (tgt_std / src_std) + tgt_mean
        src_lab[:, :, ch] = (
            src_lab[:, :, ch] * (1 - strength) + transferred * strength
        )

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def temporal_blend(
    frames: list[np.ndarray], index: int, radius: int
) -> np.ndarray:
    """Blend frame[index] with neighbors within radius for temporal smoothing."""
    if radius <= 0:
        return frames[index]

    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)
    weights = []
    stack = []
    for i in range(start, end):
        dist = abs(i - index)
        w = 1.0 / (1.0 + dist)
        weights.append(w)
        stack.append(frames[i].astype(np.float32))

    total_w = sum(weights)
    blended = sum(f * (w / total_w) for f, w in zip(stack, weights))
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_frames(
    styled_paths: list[Path],
    original_paths: list[Path],
    output_dir: Path,
    color_match_strength: float,
    temporal_blend_frames: int,
) -> list[Path]:
    """Apply color matching and temporal smoothing to styled frames."""
    output_dir.mkdir(parents=True, exist_ok=True)

    styled_bgr = [cv2.imread(str(p)) for p in styled_paths]
    original_bgr = [cv2.imread(str(p)) for p in original_paths]

    # Color transfer from originals
    for i in range(len(styled_bgr)):
        styled_bgr[i] = color_transfer(
            styled_bgr[i], original_bgr[i], color_match_strength
        )

    # Temporal smoothing
    result_paths = []
    for i in range(len(styled_bgr)):
        blended = temporal_blend(styled_bgr, i, temporal_blend_frames)
        out_path = output_dir / f"final_{styled_paths[i].stem}.png"
        cv2.imwrite(str(out_path), blended)
        result_paths.append(out_path)

        if i % 30 == 0:
            print(f"  Postprocess: {i+1}/{len(styled_bgr)}")

    return result_paths
