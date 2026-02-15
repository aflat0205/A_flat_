import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import DPTForDepthEstimation, DPTImageProcessor


def load_depth_model(model_id: str, device: str):
    """Load MiDaS depth estimation model."""
    processor = DPTImageProcessor.from_pretrained(model_id)
    model = DPTForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def estimate_depth(
    processor,
    model,
    image: Image.Image,
    device: str,
) -> Image.Image:
    """Estimate depth for a single image, return as grayscale PIL Image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = prediction.cpu().numpy()
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max - depth_min > 0:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_np = np.zeros_like(depth_np)
    return Image.fromarray(depth_np.astype(np.uint8))


def process_frames(
    model_id: str,
    device: str,
    frame_paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Generate depth maps for all frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_depth_model(model_id, device)
    depth_paths = []
    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB")
        depth_img = estimate_depth(processor, model, img, device)
        out_path = output_dir / f"depth_{fp.stem}.png"
        depth_img.save(out_path)
        depth_paths.append(out_path)
        if i % 30 == 0:
            print(f"  Depth estimation: {i+1}/{len(frame_paths)}")
    del model, processor
    torch.cuda.empty_cache()
    return depth_paths
