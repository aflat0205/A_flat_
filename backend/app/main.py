from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.v0 import feed, health
from .api.v0.avatar import jobs as avatar_jobs
from .api.v0.avatar import generate as avatar_generate
from .api.v0.avatar import assets as avatar_assets

app = FastAPI(title="A_flat_")

app.mount("/media", StaticFiles(directory="app/data"), name="media")

app.include_router(health.router, prefix="/v0")
app.include_router(feed.router, prefix="/v0")
app.include_router(avatar_assets.router, prefix="/v0/avatar")
app.include_router(avatar_generate.router, prefix="/v0/avatar")
app.include_router(avatar_jobs.router, prefix="/v0/avatar")
