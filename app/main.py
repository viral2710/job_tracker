from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.routers.api import router as api_router
from app.services import job_repository
from app.scrapers.career_site import scraper

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Career Site Job Tracker API with Swappable Matching Models, Delta Lake Performance Metrics & React UI"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount React UI Static assets
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.on_event("startup")
def startup_event():
    # Ensure default jobs exist
    if not job_repository.get_all():
        job_repository.set_jobs(scraper.SAMPLE_JOBS)

@app.get("/", include_in_schema=False)
def serve_ui():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Job Tracker API is running. Access interactive docs at /docs"}

