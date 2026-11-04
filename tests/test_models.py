import pytest
from app.models.keyword_matcher import KeywordMatcher
from app.models.tfidf_matcher import TfidfMatcher
from app.models.semantic_matcher import SemanticEmbeddingMatcher
from app.models.registry import model_registry
from app.schemas import CandidateProfile, JobPost

@pytest.fixture
def sample_jobs():
    return [
        JobPost(
            id="job_py",
            title="Senior Python Backend Engineer",
            company="Tech Corp",
            location="Remote",
            department="Engineering",
            description="Develop Python FastAPI web services and SQL databases.",
            requirements=["Python", "FastAPI", "PostgreSQL", "Docker"]
        ),
        JobPost(
            id="job_fe",
            title="Frontend React Developer",
            company="Tech Corp",
            location="New York",
            department="Engineering",
            description="Build interactive user interfaces using React and Tailwind CSS.",
            requirements=["React", "TypeScript", "Tailwind CSS", "JavaScript"]
        )
    ]

@pytest.fixture
def python_candidate():
    return CandidateProfile(
        candidate_id="cand_1",
        full_name="Py Dev",
        skills=["Python", "FastAPI", "Docker"],
        experience_summary="Experienced Python engineer building high-performance APIs.",
        preferred_roles=["Python Engineer", "Backend Engineer"],
        preferred_locations=["Remote"]
    )

def test_keyword_matcher(python_candidate, sample_jobs):
    matcher = KeywordMatcher()
    results = matcher.match(python_candidate, sample_jobs, top_k=2)
    assert len(results) == 2
    # Python job should rank higher than React frontend job
    assert results[0].job_id == "job_py"
    assert results[0].score > results[1].score
    assert "Python" in results[0].matched_skills

def test_tfidf_matcher(python_candidate, sample_jobs):
    matcher = TfidfMatcher()
    results = matcher.match(python_candidate, sample_jobs, top_k=2)
    assert len(results) == 2
    assert results[0].job_id == "job_py"
    assert results[0].score > results[1].score

def test_semantic_matcher(python_candidate, sample_jobs):
    matcher = SemanticEmbeddingMatcher()
    results = matcher.match(python_candidate, sample_jobs, top_k=2)
    assert len(results) == 2
    assert results[0].job_id == "job_py"
    assert results[0].score > 0.0

def test_model_registry():
    # Can fetch registered models
    m1 = model_registry.get_model("tfidf")
    assert m1.model_id == "tfidf"
    
    m2 = model_registry.get_model("keyword")
    assert m2.model_id == "keyword"

    m3 = model_registry.get_model("semantic")
    assert m3.model_id == "semantic"

    # Listing models
    models = model_registry.list_models()
    assert len(models) >= 3
    ids = [m.model_id for m in models]
    assert "tfidf" in ids
    assert "keyword" in ids
    assert "semantic" in ids

def test_model_registry_unknown_model():
    with pytest.raises(ValueError):
        model_registry.get_model("non_existent_model_xyz")

