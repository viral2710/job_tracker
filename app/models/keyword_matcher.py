import re
from typing import List, Set
from app.models.base import BaseJobMatcher
from app.schemas import CandidateProfile, JobPost, MatchResult

class KeywordMatcher(BaseJobMatcher):
    """
    Fast lexical matcher based on set intersection, Jaccard similarity,
    and skill keyword extraction.
    """

    @property
    def model_id(self) -> str:
        return "keyword"

    @property
    def model_name(self) -> str:
        return "Keyword / Jaccard Lexical Matcher"

    @property
    def description(self) -> str:
        return "Exact and substring skill matching with Jaccard coefficient. Extremely fast and transparent."

    @property
    def model_type(self) -> str:
        return "lexical"

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9+#.-]{2,}\b", text.lower())
        return set(tokens)

    def match(self, candidate: CandidateProfile, jobs: List[JobPost], top_k: int = 10) -> List[MatchResult]:
        if not jobs:
            return []

        # Candidate skills & tokens
        cand_skills = {s.strip().lower() for s in candidate.skills if s.strip()}
        cand_text_tokens = self._tokenize(candidate.experience_summary)
        all_candidate_tokens = cand_skills.union(cand_text_tokens)

        results: List[MatchResult] = []

        for job in jobs:
            job_text = f"{job.title} {job.department} {' '.join(job.requirements)} {job.description}".lower()
            job_tokens = self._tokenize(job_text)
            
            # Find matched explicit skills
            matched_skills = []
            for skill in candidate.skills:
                s_lower = skill.strip().lower()
                if not s_lower:
                    continue
                # Substring check or exact token check
                if s_lower in job_text or s_lower in job_tokens:
                    matched_skills.append(skill)

            # Jaccard overlap between token sets
            intersection = all_candidate_tokens.intersection(job_tokens)
            union = all_candidate_tokens.union(job_tokens)
            jaccard_score = len(intersection) / len(union) if union else 0.0

            # Skill match ratio
            skill_score = len(matched_skills) / len(cand_skills) if cand_skills else 0.0

            # Title relevance bonus
            title_tokens = self._tokenize(job.title)
            title_overlap = len(all_candidate_tokens.intersection(title_tokens))
            title_bonus = min(0.2, title_overlap * 0.1)

            # Location match bonus
            location_bonus = 0.0
            if candidate.preferred_locations:
                for pref_loc in candidate.preferred_locations:
                    if pref_loc.lower() in job.location.lower() or "remote" in job.location.lower():
                        location_bonus = 0.05
                        break

            # Composite normalized score
            composite = (0.55 * skill_score) + (0.35 * jaccard_score) + title_bonus + location_bonus
            normalized_score = round(min(1.0, max(0.0, composite)), 4)

            # Confidence level
            if normalized_score >= 0.70:
                confidence = "High"
            elif normalized_score >= 0.40:
                confidence = "Medium"
            else:
                confidence = "Low"

            # Match reasons
            reasons = []
            if matched_skills:
                reasons.append(f"Matched {len(matched_skills)} candidate skills: {', '.join(matched_skills[:4])}")
            if title_bonus > 0:
                reasons.append(f"Job title '{job.title}' aligns with candidate expertise")
            if location_bonus > 0:
                reasons.append(f"Location '{job.location}' matches preferences")
            if not reasons:
                reasons.append("General experience keyword overlap")

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

        # Sort descending by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

