from pathlib import Path
from pipelines.avatar import output_a_video


def dispatch(
    job_id: str,
    pipeline: str,
    input_video: str,
    output_video: str,
    style_id: str,
    job_dir: str,
    seed: int = 42,
):
    """Dispatch a job to the correct pipeline.

    Args:
        job_id: Unique job identifier.
        pipeline: Pipeline name ("output_a" for now).
        input_video: Path to input MP4.
        output_video: Path for output MP4.
        style_id: Style identifier.
        job_dir: Working directory for intermediates.
        seed: Random seed.
    """
    if pipeline == "output_a":
        return output_a_video.run(
            input_video=Path(input_video),
            output_video=Path(output_video),
            style_id=style_id,
            job_dir=Path(job_dir),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
