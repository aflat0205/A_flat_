import requests
from typing import Optional


class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        output_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update job status on the backend. Best-effort (no exceptions raised)."""
        try:
            payload = {"status": status}
            if progress is not None:
                payload["progress"] = progress
            if output_url:
                payload["output_url"] = output_url
            if error:
                payload["error"] = error
            requests.patch(
                f"{self.base_url}/v0/jobs/{job_id}",
                json=payload,
                timeout=5,
            )
        except Exception:
            pass  # Worker should not fail on backend comms issues
