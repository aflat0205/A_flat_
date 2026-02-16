import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def interpolate_opencv_dis(
    frame_before: Image.Image,
    frame_after: Image.Image,
    num_intermediate: int,
) -> list[Image.Image]:
    """Interpolate frames using OpenCV DIS optical flow.

    Args:
        frame_before: First keyframe
        frame_after: Second keyframe
        num_intermediate: Number of frames to generate between keyframes

    Returns:
        List of interpolated PIL Images (not including keyframes themselves)
    """
    # Convert to numpy arrays
    arr_before = np.array(frame_before)
    arr_after = np.array(frame_after)

    # Create DIS optical flow estimator
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gray_before = cv2.cvtColor(arr_before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(arr_after, cv2.COLOR_RGB2GRAY)

    # Compute optical flow from before → after
    flow = dis.calc(gray_before, gray_after, None)

    # Generate intermediate frames by warping
    interpolated = []
    h, w = arr_before.shape[:2]

    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)  # Interpolation weight (0 < t < 1)

        # Create flow map for warping
        # Scale flow by interpolation weight
        flow_map = flow * t
        # Add pixel coordinates
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

        # Warp the before frame using flow
        warped = cv2.remap(arr_before, flow_map, None, cv2.INTER_LINEAR)

        # Blend warped frame with after frame for smoother results
        blended = cv2.addWeighted(warped, 1 - t, arr_after, t, 0)
        interpolated.append(Image.fromarray(blended))

    return interpolated


def process_keyframes(
    keyframe_paths: list[Path],
    all_frame_indices: list[int],
    keyframe_interval: int,
    output_dir: Path,
) -> list[Path]:
    """Expand keyframes to full frame sequence via optical flow interpolation.

    Args:
        keyframe_paths: Paths to stylized keyframes
        all_frame_indices: Complete list of frame indices (e.g., [0, 1, 2, ..., 299])
        keyframe_interval: Spacing between keyframes (e.g., 5 for 1-in-5)
        output_dir: Directory to save full sequence

    Returns:
        List of paths to all frames (keyframes + interpolated)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all keyframes
    keyframes = [Image.open(p).convert("RGB") for p in keyframe_paths]
    all_paths = []

    # Process pairs of keyframes
    for i in range(len(keyframes) - 1):
        # Save current keyframe
        keyframe_idx = i * keyframe_interval
        out_path = output_dir / f"frame_{keyframe_idx:05d}.png"
        keyframes[i].save(out_path)
        all_paths.append(out_path)

        # Interpolate intermediate frames
        num_intermediate = keyframe_interval - 1
        if num_intermediate > 0:
            interpolated = interpolate_opencv_dis(
                keyframes[i], keyframes[i + 1], num_intermediate
            )
            for j, img in enumerate(interpolated):
                idx = keyframe_idx + j + 1
                out_path = output_dir / f"frame_{idx:05d}.png"
                img.save(out_path)
                all_paths.append(out_path)

    # Save final keyframe
    final_keyframe_idx = (len(keyframes) - 1) * keyframe_interval
    out_path = output_dir / f"frame_{final_keyframe_idx:05d}.png"
    keyframes[-1].save(out_path)
    all_paths.append(out_path)

    print(
        f"  Interpolated {len(all_paths)} frames from {len(keyframes)} keyframes (1:{keyframe_interval})"
    )
    return all_paths
