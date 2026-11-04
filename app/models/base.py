from abc import ABC, abstractmethod
from typing import List
from app.schemas import CandidateProfile, JobPost, MatchResult

class BaseJobMatcher(ABC):
    """
    Abstract base strategy for job matching models.
    Allows easy swapping between algorithms (TF-IDF, Keywords, Embeddings, LLMs).
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for the model (e.g. 'tfidf', 'keyword')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief summary of the algorithm/architecture."""
        pass

    @property
    def model_type(self) -> str:
        """Category: 'lexical', 'statistical', 'semantic', 'llm'."""
        return "general"

    @abstractmethod
    def match(self, candidate: CandidateProfile, jobs: List[JobPost], top_k: int = 10) -> List[MatchResult]:
        """
        Rank candidate against a list of jobs.
        Returns a list of MatchResult sorted by descending score.
        """
        pass

