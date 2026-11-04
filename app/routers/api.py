from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.schemas import CandidateProfile, JobPost, MatchResponse, ModelInfo
from app.services import matching_service, job_repository
from app.scrapers.career_site import scraper
from app.models.registry import model_registry
from app.metrics.delta_logger import metrics_logger

router = APIRouter(prefix="/api/v1", tags=["Job Tracker"])

class ScrapeRequest(BaseModel):
    url: str = Field(default="sample", description="Career site URL or 'sample' for demo positions")
    replace_existing: bool = Field(default=False, description="Whether to replace current job store")

class ScrapeResponse(BaseModel):
    message: str
    jobs_fetched: int
    source_url: str

class MatchRequest(BaseModel):
    candidate: CandidateProfile
    model_name: Optional[str] = Field(default=None, description="Model ID: 'tfidf', 'keyword', or 'semantic'")
    top_k: Optional[int] = Field(default=10, ge=1, le=50)

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all registered matching models, their descriptions, and default status."""
    return model_registry.list_models()

@router.get("/jobs", response_model=List[JobPost])
def get_jobs():
    """List all currently ingested job postings."""
    jobs = job_repository.get_all()
    if not jobs:
        # Auto-seed sample jobs if repository is empty
        jobs = scraper.SAMPLE_JOBS
        job_repository.set_jobs(jobs)
    return jobs

@router.post("/jobs", response_model=JobPost)
def create_job(job: JobPost):
    """Add a new custom job posting to the repository."""
    job_repository.add_jobs([job])
    return job

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_career_site(payload: ScrapeRequest):
    """Fetch/scrape jobs from a target career site URL and ingest them."""
    fetched = await scraper.fetch_jobs(payload.url)
    if payload.replace_existing:
        job_repository.set_jobs(fetched)
    else:
        job_repository.add_jobs(fetched)

    return ScrapeResponse(
        message=f"Successfully ingested {len(fetched)} jobs from career site.",
        jobs_fetched=len(fetched),
        source_url=payload.url
    )

@router.post("/match", response_model=MatchResponse)
def match_candidate(payload: MatchRequest):
    """
    Match candidate resume/skills against active career postings.
    Model can be easily selected via payload.model_name ('tfidf', 'keyword', 'semantic').
    Automatically records latency and score metrics to Delta Lake table.
    """
    try:
        return matching_service.match_jobs(
            candidate=payload.candidate,
            model_id=payload.model_name,
            top_k=payload.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics/summary")
def get_metrics_summary():
    """Get aggregated performance metrics grouped by model from the Delta Lake table."""
    return metrics_logger.get_summary_by_model()

@router.get("/metrics/history")
def get_metrics_history(limit: int = Query(default=50, ge=1, le=500)):
    """Fetch raw execution metrics audit log from Delta Lake table."""
    records = metrics_logger.get_all_metrics()
    return list(reversed(records))[:limit]

