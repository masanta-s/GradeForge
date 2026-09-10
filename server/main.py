"""GradeForge API.

    . .\\env.ps1
    uvicorn server.main:app --port 8000

Serves the React app's production build from frontend/dist when it exists; during development
run Vite separately (it proxies /api to this server).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import config  # noqa: F401  (redirects caches/temp; offline HuggingFace)
from server.routes import exams, sheets, system
from server.services import Services

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class SPAStaticFiles(StaticFiles):
    """Serve the React build; unknown non-API paths get index.html so client-side routes
    (e.g. a refresh on /exams/abc) load the app instead of a 404."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code == 404 and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.services.jobs.shutdown()

    app = FastAPI(title="GradeForge", version="0.1.0", lifespan=lifespan)
    app.state.services = services or Services()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
        allow_methods=["*"], allow_headers=["*"],
    )
    for module in (system, exams, sheets):
        app.include_router(module.router)
    if FRONTEND_DIST.is_dir():
        app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


app = create_app()
