import torch
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from pipelines.avatar.style_config import StyleConfig


def load_pipeline(style: StyleConfig, device: str, dtype_str: str):
    """Load SD1.5 + dual ControlNet pipeline for the given style."""
    dtype = torch.float16 if dtype_str == "float16" else torch.float32

    controlnet_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=dtype,
    )
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        style.model_id,
        controlnet=[controlnet_openpose, controlnet_depth],
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def stylize_frame(
    pipe,
    style: StyleConfig,
    source_image: Image.Image,
    openpose_image: Image.Image,
    depth_image: Image.Image,
    seed: int = 42,
) -> Image.Image:
    """Apply style to a single frame using img2img + ControlNet."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    prompt = style.prompt_template.replace("{face_description}", "person face")

    result = pipe(
        prompt=prompt,
        negative_prompt=style.negative_prompt,
        image=source_image,
        control_image=[openpose_image, depth_image],
        num_inference_steps=style.num_inference_steps,
        guidance_scale=style.guidance_scale,
        strength=style.denoising_strength,
        controlnet_conditioning_scale=[
            style.controlnet_openpose_weight,
            style.controlnet_depth_weight,
        ],
        generator=generator,
    )
    return result.images[0]


def process_frames(
    style: StyleConfig,
    device: str,
    dtype: str,
    frame_paths: list[Path],
    openpose_paths: list[Path],
    depth_paths: list[Path],
    output_dir: Path,
    seed: int = 42,
) -> list[Path]:
    """Stylize all frames. Returns paths to styled frame images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(style, device, dtype)
    styled_paths = []

    for i, (fp, op, dp) in enumerate(zip(frame_paths, openpose_paths, depth_paths)):
        src = Image.open(fp).convert("RGB")
        pose = Image.open(op).convert("RGB")
        depth = Image.open(dp).convert("RGB")

        styled = stylize_frame(pipe, style, src, pose, depth, seed=seed)
        out_path = output_dir / f"styled_{fp.stem}.png"
        styled.save(out_path)
        styled_paths.append(out_path)

        if i % 10 == 0:
            print(f"  Stylize: {i+1}/{len(frame_paths)}")

    del pipe
    torch.cuda.empty_cache()
    return styled_paths
