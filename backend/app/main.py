from fastapi import FastAPI
from .api.v0 import feed, health

app = FastAPI(title="A_flat_")

app.include_router(health.router, prefix="/v0")
app.include_router(feed.router, prefix="/v0")
