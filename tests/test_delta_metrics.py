import shutil
from pathlib import Path
import pytest
from app.metrics.delta_logger import DeltaMetricsLogger
from app.schemas import PerformanceMetricRecord

@pytest.fixture
def temp_delta_logger(tmp_path):
    table_dir = tmp_path / "test_delta_metrics"
    logger = DeltaMetricsLogger(table_path=table_dir)
    return logger

def test_delta_logging_and_retrieval(temp_delta_logger):
    assert not temp_delta_logger.is_table_initialized()

    # Log record 1
    r1 = PerformanceMetricRecord(
        run_id="run_001",
        timestamp="2026-11-04T12:00:00Z",
        model_name="tfidf",
        candidate_id="cand_alex",
        num_jobs_evaluated=10,
        latency_ms=12.5,
        top_score=0.88,
        mean_score=0.45,
        top_job_id="job_001",
        top_job_title="Senior Python Backend Engineer"
    )
    temp_delta_logger.log_metric(r1)

    assert temp_delta_logger.is_table_initialized()

    # Log record 2 (keyword model)
    r2 = PerformanceMetricRecord(
        run_id="run_002",
        timestamp="2026-11-04T12:05:00Z",
        model_name="keyword",
        candidate_id="cand_alex",
        num_jobs_evaluated=10,
        latency_ms=3.2,
        top_score=0.75,
        mean_score=0.35,
        top_job_id="job_001",
        top_job_title="Senior Python Backend Engineer"
    )
    temp_delta_logger.log_metric(r2)

    all_metrics = temp_delta_logger.get_all_metrics()
    assert len(all_metrics) == 2
    models_logged = {m["model_name"] for m in all_metrics}
    assert models_logged == {"tfidf", "keyword"}

    # Test Summary aggregation
    summary = temp_delta_logger.get_summary_by_model()
    assert len(summary) == 2
    tfidf_sum = next(s for s in summary if s["model_name"] == "tfidf")
    assert tfidf_sum["total_runs"] == 1
    assert tfidf_sum["avg_latency_ms"] == 12.5

