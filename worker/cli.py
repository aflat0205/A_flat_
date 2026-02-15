
import argparse
import uuid
from pathlib import Path

from pipelines.avatar.run_job import dispatch
from pipelines.avatar.style_config import STYLES


def main():
    parser = argparse.ArgumentParser(description="Output A Pipeline CLI")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input MP4 video"
    )
    parser.add_argument(
        "--style", "-s", required=True,
        choices=list(STYLES.keys()) + ["all"],
        help="Style ID or 'all' to run all styles",
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/outputs",
        help="Output directory (default: data/outputs)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    styles_to_run = list(STYLES.keys()) if args.style == "all" else [args.style]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for style_id in styles_to_run:
        job_id = str(uuid.uuid4())[:8]
        job_dir = output_dir / f"job_{job_id}_{style_id}"
        output_path = output_dir / f"output_{style_id}_{input_path.stem}.mp4"

        print(f"\n{'=' * 60}")
        print(f"Style: {style_id}")
        print(f"Job dir: {job_dir}")
        print(f"Output: {output_path}")
        print(f"{'=' * 60}\n")

        dispatch(
            job_id=job_id,
            pipeline="output_a",
            input_video=str(input_path),
            output_video=str(output_path),
            style_id=style_id,
            job_dir=str(job_dir),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
