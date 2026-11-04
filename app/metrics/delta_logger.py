import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from app.config import settings
from app.schemas import PerformanceMetricRecord

class DeltaMetricsLogger:
    """
    Persists and queries model performance metrics using Delta Lake format.
    Stores latency, score distributions, and top matches partitioned or tracked by model_name.
    """

    def __init__(self, table_path: Optional[Path] = None):
        self.table_path = table_path or settings.delta_metrics_path
        self.table_path.mkdir(parents=True, exist_ok=True)

        # PyArrow Schema for Delta Lake table
        self.schema = pa.schema([
            ("run_id", pa.string()),
            ("timestamp", pa.string()),
            ("model_name", pa.string()),
            ("candidate_id", pa.string()),
            ("num_jobs_evaluated", pa.int64()),
            ("latency_ms", pa.float64()),
            ("top_score", pa.float64()),
            ("mean_score", pa.float64()),
            ("top_job_id", pa.string()),
            ("top_job_title", pa.string())
        ])

    def log_metric(self, record: PerformanceMetricRecord) -> None:
        """Appends a new performance record to the Delta Lake table."""
        data = {
            "run_id": [record.run_id],
            "timestamp": [record.timestamp],
            "model_name": [record.model_name],
            "candidate_id": [record.candidate_id],
            "num_jobs_evaluated": [record.num_jobs_evaluated],
            "latency_ms": [record.latency_ms],
            "top_score": [record.top_score],
            "mean_score": [record.mean_score],
            "top_job_id": [record.top_job_id or ""],
            "top_job_title": [record.top_job_title or ""]
        }
        df = pd.DataFrame(data)
        
        # Convert DataFrame to PyArrow Table using strict schema
        pa_table = pa.Table.from_pandas(df, schema=self.schema)

        # Write to Delta Lake table (mode="append")
        write_deltalake(
            str(self.table_path),
            pa_table,
            mode="append"
        )

    def is_table_initialized(self) -> bool:
        """Check if delta table directory contains a valid Delta table."""
        try:
            DeltaTable(str(self.table_path))
            return True
        except Exception:
            return False

    def get_all_metrics(self) -> List[dict]:
        """Reads all metric records from the Delta Table as a list of dicts."""
        if not self.is_table_initialized():
            return []
        
        dt = DeltaTable(str(self.table_path))
        df = dt.to_pandas()
        return df.to_dict(orient="records")

    def get_summary_by_model(self) -> List[dict]:
        """
        Aggregates performance statistics grouped by model_name:
        - total_runs
        - avg_latency_ms
        - avg_top_score
        - avg_mean_score
        - last_run_timestamp
        """
        if not self.is_table_initialized():
            return []

        dt = DeltaTable(str(self.table_path))
        df = dt.to_pandas()
        if df.empty:
            return []

        summary = []
        for model_name, group in df.groupby("model_name"):
            summary.append({
                "model_name": str(model_name),
                "total_runs": int(len(group)),
                "avg_latency_ms": round(float(group["latency_ms"].mean()), 2),
                "min_latency_ms": round(float(group["latency_ms"].min()), 2),
                "max_latency_ms": round(float(group["latency_ms"].max()), 2),
                "avg_top_score": round(float(group["top_score"].mean()), 3),
                "avg_mean_score": round(float(group["mean_score"].mean()), 3),
                "last_run": str(group["timestamp"].max())
            })
        return summary

metrics_logger = DeltaMetricsLogger()
