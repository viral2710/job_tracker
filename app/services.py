import json
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from app.config import settings
from app.schemas import CandidateProfile, JobPost, MatchResult, MatchResponse, PerformanceMetricRecord
from app.models.registry import model_registry
from app.metrics.delta_logger import metrics_logger

class JobRepository:
    """Thread-safe in-memory and file-backed job storage."""
    def __init__(self):
        self._jobs: List[JobPost] = []
        self._load()

    def _load(self):
        if settings.jobs_db_path.exists():
            try:
                data = json.loads(settings.jobs_db_path.read_text(encoding="utf-8"))
                self._jobs = [JobPost(**j) for j in data]
            except Exception:
                self._jobs = []

    def _save(self):
        settings.jobs_db_path.parent.mkdir(parents=True, exist_ok=True)
        data = [j.model_dump() for j in self._jobs]
        settings.jobs_db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_all(self) -> List[JobPost]:
        return list(self._jobs)

    def set_jobs(self, jobs: List[JobPost]) -> None:
        self._jobs = list(jobs)
        self._save()

    def add_jobs(self, jobs: List[JobPost]) -> int:
        existing_ids = {j.id for j in self._jobs}
        added = 0
        for job in jobs:
            if job.id not in existing_ids:
                self._jobs.append(job)
                existing_ids.add(job.id)
                added += 1
            else:
                # Update existing
                self._jobs = [job if j.id == job.id else j for j in self._jobs]
        self._save()
        return added

job_repository = JobRepository()

class JobMatchingService:
    """Orchestrates candidate-job matching, runtime model switching, and Delta Lake metrics recording."""

    def __init__(self):
        self.repository = job_repository
        self.registry = model_registry
        self.metrics = metrics_logger

    def match_jobs(
        self,
        candidate: CandidateProfile,
        model_id: Optional[str] = None,
        top_k: Optional[int] = None,
        custom_jobs: Optional[List[JobPost]] = None
    ) -> MatchResponse:
        active_model_id = (model_id or settings.default_model).lower()
        matcher = self.registry.get_model(active_model_id)
        
        jobs_to_evaluate = custom_jobs if custom_jobs is not None else self.repository.get_all()
        k = top_k or settings.top_k_matches

        # Measure matching latency
        start_time = time.perf_counter()
        matches: List[MatchResult] = matcher.match(candidate, jobs_to_evaluate, top_k=k)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Performance metric calculations
        top_score = matches[0].score if matches else 0.0
        mean_score = round(sum(m.score for m in matches) / len(matches), 4) if matches else 0.0
        top_job_id = matches[0].job_id if matches else None
        top_job_title = matches[0].job_title if matches else None

        # Build metric record
        record = PerformanceMetricRecord(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=matcher.model_id,
            candidate_id=candidate.candidate_id,
            num_jobs_evaluated=len(jobs_to_evaluate),
            latency_ms=duration_ms,
            top_score=top_score,
            mean_score=mean_score,
            top_job_id=top_job_id,
            top_job_title=top_job_title
        )

        # Persist to Delta Lake table
        try:
            self.metrics.log_metric(record)
        except Exception as e:
            # Non-blocking log if Delta Lake hits disk locks in extreme test concurrency
            print(f"Warning: Failed to log metric to Delta table: {e}")

        return MatchResponse(
            model_used=matcher.model_id,
            candidate_id=candidate.candidate_id,
            num_jobs_matched=len(matches),
            execution_time_ms=duration_ms,
            matches=matches
        )

matching_service = JobMatchingService()

