import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from pathlib import Path

_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION


def create_landmarker(model_path: Path) -> FaceLandmarker:
    """Create MediaPipe FaceLandmarker instance."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def detect_and_render(
    landmarker: FaceLandmarker,
    frame_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> bool:
    """Detect face landmarks and render openpose-style image.
    Returns True if face was detected."""
    mp_image = mp.Image.create_from_file(str(frame_path))
    result = landmarker.detect(mp_image)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if not result.face_landmarks:
        cv2.imwrite(str(output_path), canvas)
        return False

    landmarks = result.face_landmarks[0]
    # Draw face mesh connections
    for connection in _FACE_CONNECTIONS:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        pt1 = (int(start.x * width), int(start.y * height))
        pt2 = (int(end.x * width), int(end.y * height))
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 1)

    # Key landmark points with colored dots
    KEY_POINTS = {
        1: (0, 255, 0),      # nose tip
        33: (255, 0, 0),     # left eye inner
        263: (255, 0, 0),    # right eye inner
        61: (0, 0, 255),     # left mouth corner
        291: (0, 0, 255),    # right mouth corner
        10: (255, 255, 0),   # forehead
        152: (255, 255, 0),  # chin
    }
    for idx, color in KEY_POINTS.items():
        lm = landmarks[idx]
        pt = (int(lm.x * width), int(lm.y * height))
        cv2.circle(canvas, pt, 3, color, -1)

    cv2.imwrite(str(output_path), canvas)
    return True


def process_frames(
    model_path: Path,
    frame_paths: list[Path],
    output_dir: Path,
    width: int,
    height: int,
) -> list[Path]:
    """Process all frames, return paths to openpose-style renders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    landmarker = create_landmarker(model_path)
    pose_paths = []
    for i, fp in enumerate(frame_paths):
        out_path = output_dir / f"pose_{fp.stem}.png"
        detect_and_render(landmarker, fp, out_path, width, height)
        pose_paths.append(out_path)
        if i % 30 == 0:
            print(f"  Face landmarks: {i+1}/{len(frame_paths)}")
    landmarker.close()
    return pose_paths
