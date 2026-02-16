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

# MediaPipe face mesh tessellation connections
# Source: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
_FACE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356),
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149),
    (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10), (10, 338), (297, 338), (332, 297), (284, 332), (251, 284), (389, 251),
    (356, 389), (454, 356), (323, 454), (361, 323), (288, 361), (397, 288), (365, 397),
    (379, 365), (378, 379), (400, 378), (377, 400), (152, 377), (148, 152), (176, 148),
    (149, 176), (150, 149), (136, 150), (172, 136), (58, 172), (132, 58), (93, 132),
    (234, 93), (127, 234), (162, 127), (21, 162), (54, 21), (103, 54), (67, 103),
    (109, 67), (10, 109), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365),
    (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93),
    (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67),
    (67, 109), (109, 10), (338, 10), (297, 338), (332, 297), (284, 332)
])


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
