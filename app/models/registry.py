from typing import Dict, List, Optional
from app.models.base import BaseJobMatcher
from app.models.keyword_matcher import KeywordMatcher
from app.models.tfidf_matcher import TfidfMatcher
from app.models.semantic_matcher import SemanticEmbeddingMatcher
from app.models.gemini_matcher import GeminiJobMatcher
from app.schemas import ModelInfo
from app.config import settings

class ModelRegistry:
    """
    Central registry for job matching models.
    Supports runtime retrieval, dynamic registration of new models,
    and inspection of available models.
    """

    def __init__(self):
        self._models: Dict[str, BaseJobMatcher] = {}
        self._register_default_models()

    def _register_default_models(self):
        # Register standard statistical & semantic models
        self.register(KeywordMatcher())
        self.register(TfidfMatcher())
        self.register(SemanticEmbeddingMatcher())
        
        # Register configured Gemini LLM model
        self.register(GeminiJobMatcher(model_variant=settings.gemini_model))

        # Register top benchmark Gemini alternatives for live comparisons
        self.register(GeminiJobMatcher(model_variant="gemini-3.8-flash"))
        self.register(GeminiJobMatcher(model_variant="gemini-2.5-flash"))

    def register(self, model: BaseJobMatcher):
        """Register a new matcher instance."""
        self._models[model.model_id.lower()] = model

    def get_model(self, model_id: str) -> BaseJobMatcher:
        """Fetch model by ID, raising ValueError if not found."""
        m_id = model_id.lower()
        if m_id in self._models:
            return self._models[m_id]

        # Dynamic instantiation for arbitrary Gemini models passed by user
        if m_id.startswith("gemini_") or m_id.startswith("models/"):
            clean_name = m_id.replace("gemini_", "").replace("models/", "").replace("_", "-")
            custom_gemini = GeminiJobMatcher(model_variant=clean_name)
            self.register(custom_gemini)
            return custom_gemini

        available = ", ".join(self._models.keys())
        raise ValueError(f"Model '{model_id}' not found. Available models: {available}")

    def list_models(self, default_id: Optional[str] = None) -> List[ModelInfo]:
        """List metadata for all registered models."""
        active_default = default_id or settings.default_model or "gemini"
        return [
            ModelInfo(
                model_id=m.model_id,
                name=m.model_name,
                description=m.description,
                type=m.model_type,
                is_default=(m.model_id == active_default)
            )
            for m in self._models.values()
        ]

# Global singleton registry
model_registry = ModelRegistry()
