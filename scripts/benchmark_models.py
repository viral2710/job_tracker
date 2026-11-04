import time
from app.models.registry import model_registry
from app.models.gemini_matcher import GeminiJobMatcher
from app.schemas import CandidateProfile
from app.scrapers.career_site import scraper
from app.metrics.delta_logger import metrics_logger
from app.services import matching_service

def benchmark_gemini_models_to_delta():
    print("=" * 80)
    print("  BENCHMARKING ACTIVE GEMINI MODELS & COMMITTING TO DELTA LAKE")
    print("=" * 80)

    candidate = CandidateProfile(
        candidate_id="cand_alex_gemini_bench",
        full_name="Alex Rivera",
        skills=["Python", "FastAPI", "Docker", "Delta Lake", "REST APIs", "PostgreSQL"],
        experience_summary="Lead Software Engineer with 6 years designing Python backend microservices using FastAPI, event-driven architectures, and Delta Lake analytics pipelines.",
        preferred_roles=["Senior Python Engineer", "Backend Architect"],
        preferred_locations=["Remote"]
    )

    jobs = scraper.SAMPLE_JOBS[:4]

    # Models to test and compare
    models_to_evaluate = [
        ("tfidf", "TF-IDF + Cosine"),
        ("keyword", "Keyword / Jaccard"),
        ("semantic", "Dense Semantic"),
        ("gemini_3_1_flash_lite", "Gemini 3.1 Flash Lite"),
        ("gemini_3_flash_preview", "Gemini 3 Flash Preview"),
        ("gemini_3_5_flash", "Gemini 3.5 Flash"),
    ]

    for model_id, label in models_to_evaluate:
        print(f"\n[+] Evaluating Model: {label} ({model_id})...")
        try:
            res = matching_service.match_jobs(
                candidate=candidate,
                model_id=model_id,
                top_k=3,
                custom_jobs=jobs
            )
            top = res.matches[0] if res.matches else None
            top_title = top.job_title if top else "N/A"
            top_score = f"{round(top.score * 100, 1)}%" if top else "0%"
            print(f"    -> Latency: {res.execution_time_ms} ms | Top Match: '{top_title}' ({top_score} - {top.confidence})")
        except Exception as e:
            print(f"    -> Evaluation failed: {e}")

    print("\n" + "=" * 80)
    print("  DELTA LAKE SUMMARY ACROSS ALL BENCHMARKED MODELS")
    print("=" * 80)
    summary = metrics_logger.get_summary_by_model()
    for s in summary:
        print(f" - {s['model_name']:<25} | Runs: {s['total_runs']} | Avg Latency: {s['avg_latency_ms']:>8} ms | Avg Top Score: {s['avg_top_score']}")

if __name__ == "__main__":
    benchmark_gemini_models_to_delta()
