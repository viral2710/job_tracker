import json
import re
from typing import List, Optional
import httpx
from app.models.base import BaseJobMatcher
from app.schemas import CandidateProfile, JobPost, MatchResult
from app.config import settings

class GeminiJobMatcher(BaseJobMatcher):
    """
    Job Matcher powered by Google Gemini LLM via direct REST API.
    Dynamically configurable with different Gemini models (e.g. gemini-3.5-flash-lite, gemini-3.8-flash, etc.).
    Extracts deep reasoning, skill overlap, and semantic suitability scores.
    """

    def __init__(self, model_variant: Optional[str] = None):
        self.target_model = model_variant or settings.gemini_model or "gemini-3.5-flash-lite"
        if self.target_model.startswith("models/"):
            self.target_model = self.target_model[len("models/"):]

    @property
    def model_id(self) -> str:
        if self.target_model in ["gemini-3.5-flash-lite", settings.gemini_model]:
            return "gemini"
        clean = self.target_model.replace(".", "_").replace("-", "_")
        return f"gemini_{clean}"

    @property
    def model_name(self) -> str:
        return f"Google Gemini LLM ({self.target_model})"

    @property
    def description(self) -> str:
        return f"Deep contextual reasoning & evaluation using Google Gemini frontier LLM ({self.target_model})."

    @property
    def model_type(self) -> str:
        return "llm"

    def _build_prompt(self, candidate: CandidateProfile, jobs: List[JobPost]) -> str:
        jobs_data = []
        for j in jobs:
            jobs_data.append({
                "job_id": j.id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "requirements": j.requirements,
                "description_snippet": j.description[:200]
            })

        candidate_data = {
            "name": candidate.full_name,
            "skills": candidate.skills,
            "experience": candidate.experience_summary[:300],
            "preferred_roles": candidate.preferred_roles,
            "preferred_locations": candidate.preferred_locations
        }

        return f"""You are an expert AI Technical Recruiter.
Evaluate candidate against available jobs.
Output JSON only with list of evaluations containing job_id, score (0.00-1.00), confidence (High/Medium/Low), matched_skills, and 1 concise reason.

Candidate:
{json.dumps(candidate_data)}

Jobs:
{json.dumps(jobs_data)}

Format:
{{"evaluations": [{{"job_id": "string", "score": 0.85, "confidence": "High", "matched_skills": ["Python"], "reasons": ["Relevant backend background"]}}]}}
"""

    def match(self, candidate: CandidateProfile, jobs: List[JobPost], top_k: int = 10) -> List[MatchResult]:
        if not jobs:
            return []

        api_key = settings.gemini_api_key
        if not api_key:
            return self._fallback_match(candidate, jobs, "Gemini API key not configured in .env", top_k)

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.target_model}:generateContent?key={api_key}"
        prompt = self._build_prompt(candidate, jobs)

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(endpoint, json=payload)

            if res.status_code != 200:
                return self._fallback_match(candidate, jobs, f"Gemini API HTTP {res.status_code}: {res.text[:80]}", top_k)

            res_json = res.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                return self._fallback_match(candidate, jobs, "No candidate output from Gemini", top_k)

            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            clean_json = raw_text.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\n", "", clean_json)
                clean_json = re.sub(r"\n```$", "", clean_json)

            data = json.loads(clean_json)
            evals = {e.get("job_id"): e for e in data.get("evaluations", [])}

            results: List[MatchResult] = []
            for job in jobs:
                eval_item = evals.get(job.id, {})
                score = float(eval_item.get("score", 0.5))
                confidence = eval_item.get("confidence", "Medium")
                matched_skills = eval_item.get("matched_skills", [])
                reasons = eval_item.get("reasons", [f"Evaluated with Gemini {self.target_model}"])

                results.append(
                    MatchResult(
                        job_id=job.id,
                        job_title=job.title,
                        company=job.company,
                        location=job.location,
                        score=round(min(1.0, max(0.0, score)), 4),
                        confidence=confidence,
                        matched_skills=matched_skills,
                        match_reasons=reasons,
                        job_url=job.url
                    )
                )

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            return self._fallback_match(candidate, jobs, f"Gemini LLM call failed: {str(e)[:60]}", top_k)

    def _fallback_match(self, candidate: CandidateProfile, jobs: List[JobPost], reason: str, top_k: int) -> List[MatchResult]:
        """Provides heuristic fallback if Gemini API is unreachable or rate-limited."""
        from app.models.tfidf_matcher import TfidfMatcher
        fallback = TfidfMatcher().match(candidate, jobs, top_k=top_k)
        for r in fallback:
            r.match_reasons.insert(0, f"Fallback mode: {reason}")
        return fallback
