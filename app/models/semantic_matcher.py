import re
import math
from typing import List, Dict
import numpy as np
from app.models.base import BaseJobMatcher
from app.schemas import CandidateProfile, JobPost, MatchResult

class SemanticEmbeddingMatcher(BaseJobMatcher):
    """
    Semantic matcher using pseudo-dense character n-gram / hashing projection embeddings
    with Cosine Similarity. Designed to provide semantic soft matching with zero
    network download requirements, fully swappable with transformer embeddings (e.g. sentence-transformers).
    """

    def __init__(self, embedding_dim: int = 128):
        self.dim = embedding_dim

    @property
    def model_id(self) -> str:
        return "semantic"

    @property
    def model_name(self) -> str:
        return "Semantic Dense Projection Matcher"

    @property
    def description(self) -> str:
        return "Dense multi-token semantic embedding projection. Captures contextual nuances and fuzzy terminology."

    @property
    def model_type(self) -> str:
        return "semantic"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return vec

        for word in words:
            # Hash word and character 3-grams to project into dense space
            h = hash(word)
            idx = abs(h) % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign

            for i in range(len(word) - 2):
                tri_hash = hash(word[i:i+3])
                tri_idx = abs(tri_hash) % self.dim
                tri_sign = 1.0 if (tri_hash // self.dim) % 2 == 0 else -1.0
                vec[tri_idx] += 0.5 * tri_sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def match(self, candidate: CandidateProfile, jobs: List[JobPost], top_k: int = 10) -> List[MatchResult]:
        if not jobs:
            return []

        cand_text = f"{' '.join(candidate.skills)} {candidate.experience_summary}"
        cand_vec = self._embed(cand_text)

        results: List[MatchResult] = []

        for job in jobs:
            job_text = f"{job.title} {job.department} {' '.join(job.requirements)} {job.description}"
            job_vec = self._embed(job_text)

            sim = float(np.dot(cand_vec, job_vec))
            # Rescale cosine (-1 to 1) to (0 to 1)
            normalized_score = round(max(0.0, min(1.0, (sim + 1.0) / 2.0)), 4)

            # Matched explicit skills
            job_text_lower = job_text.lower()
            matched_skills = [
                s for s in candidate.skills if s.lower() in job_text_lower
            ]

            if normalized_score >= 0.70:
                confidence = "High"
            elif normalized_score >= 0.45:
                confidence = "Medium"
            else:
                confidence = "Low"

            reasons = [
                f"Dense semantic similarity: {round(normalized_score * 100, 1)}%",
                f"Evaluated {len(job.requirements)} core requirements"
            ]
            if matched_skills:
                reasons.append(f"Matching candidate skills: {', '.join(matched_skills[:3])}")

            results.append(
                MatchResult(
                    job_id=job.id,
                    job_title=job.title,
                    company=job.company,
                    location=job.location,
                    score=normalized_score,
                    confidence=confidence,
                    matched_skills=matched_skills,
                    match_reasons=reasons,
                    job_url=job.url
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

