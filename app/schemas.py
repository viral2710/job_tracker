from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class JobPost(BaseModel):
    id: str
    title: str
    company: str
    location: str
    department: Optional[str] = "Engineering"
    employment_type: Optional[str] = "Full-time"
    description: str
    requirements: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    source: Optional[str] = "career_site"
    posted_date: Optional[str] = None

class CandidateProfile(BaseModel):
    candidate_id: str = "cand_default"
    full_name: Optional[str] = "Candidate"
    skills: List[str] = Field(default_factory=list)
    experience_summary: str
    preferred_roles: List[str] = Field(default_factory=list)
    preferred_locations: List[str] = Field(default_factory=list)

class MatchResult(BaseModel):
    job_id: str
    job_title: str
    company: str
    location: str
    score: float = Field(..., description="Normalized score between 0.0 and 1.0")
    confidence: str = Field(..., description="High, Medium, or Low")
    matched_skills: List[str] = Field(default_factory=list)
    match_reasons: List[str] = Field(default_factory=list)
    job_url: Optional[str] = None

class MatchResponse(BaseModel):
    model_used: str
    candidate_id: str
    num_jobs_matched: int
    execution_time_ms: float
    matches: List[MatchResult]

class ModelInfo(BaseModel):
    model_id: str
    name: str
    description: str
    type: str
    is_default: bool = False

class PerformanceMetricRecord(BaseModel):
    run_id: str
    timestamp: str
    model_name: str
    candidate_id: str
    num_jobs_evaluated: int
    latency_ms: float
    top_score: float
    mean_score: float
    top_job_id: Optional[str] = None
    top_job_title: Optional[str] = None

