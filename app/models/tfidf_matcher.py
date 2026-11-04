from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.models.base import BaseJobMatcher
from app.schemas import CandidateProfile, JobPost, MatchResult

class TfidfMatcher(BaseJobMatcher):
    """
    Statistical matcher leveraging TF-IDF vectorization with 1-2 n-grams,
    sublinear term frequency scaling, and cosine similarity.
    """

    @property
    def model_id(self) -> str:
        return "tfidf"

    @property
    def model_name(self) -> str:
        return "TF-IDF + Cosine Similarity Matcher"

    @property
    def description(self) -> str:
        return "Sublinear TF-IDF vectorization with n-grams and cosine distance. Strong statistical IR baseline."

    @property
    def model_type(self) -> str:
        return "statistical"

    def match(self, candidate: CandidateProfile, jobs: List[JobPost], top_k: int = 10) -> List[MatchResult]:
        if not jobs:
            return []

        # Build candidate document representation with skill weighting
        cand_skills_str = " ".join([f"{s} {s}" for s in candidate.skills])  # double weight
        candidate_doc = f"{cand_skills_str} {candidate.experience_summary}"

        # Build job documents with weighted title and requirements
        job_docs = []
        for job in jobs:
            title_weighted = f"{job.title} {job.title} {job.title}"
            reqs_weighted = " ".join(job.requirements)
            doc = f"{title_weighted} {job.department} {reqs_weighted} {job.description}"
            job_docs.append(doc)

        corpus = [candidate_doc] + job_docs

        # Vectorize corpus
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=5000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            cand_vector = tfidf_matrix[0:1]
            job_vectors = tfidf_matrix[1:]

            # Compute cosine similarity
            similarities = cosine_similarity(cand_vector, job_vectors)[0]
        except Exception:
            # Fallback if vocabulary is completely empty
            similarities = np.zeros(len(jobs))

        cand_skills_lower = {s.lower() for s in candidate.skills}
        results: List[MatchResult] = []

        for idx, job in enumerate(jobs):
            raw_score = float(similarities[idx])
            job_text_lower = f"{job.title} {' '.join(job.requirements)} {job.description}".lower()

            # Matched candidate skills
            matched_skills = [
                s for s in candidate.skills if s.lower() in job_text_lower
            ]

            # Boost slightly for explicit matched skills
            skill_bonus = min(0.15, (len(matched_skills) / max(1, len(candidate.skills))) * 0.15) if candidate.skills else 0.0
            final_score = round(min(1.0, max(0.0, raw_score + skill_bonus)), 4)

            # Confidence
            if final_score >= 0.65:
                confidence = "High"
            elif final_score >= 0.35:
                confidence = "Medium"
            else:
                confidence = "Low"

            # Reasons
            reasons = [f"TF-IDF similarity: {round(raw_score * 100, 1)}%"]
            if matched_skills:
                reasons.append(f"Identified skills: {', '.join(matched_skills[:4])}")
            if any(role.lower() in job.title.lower() for role in candidate.preferred_roles):
                reasons.append(f"Role title '{job.title}' closely matches preferred target role")

            results.append(
                MatchResult(
                    job_id=job.id,
                    job_title=job.title,
                    company=job.company,
                    location=job.location,
                    score=final_score,
                    confidence=confidence,
                    matched_skills=matched_skills,
                    match_reasons=reasons,
                    job_url=job.url
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

