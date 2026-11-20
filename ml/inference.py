"""
Telecom QoS Anomaly Detection - Inference Module
Owned by: Person 1 (ML & Computer Vision Engineer)
Consumer: Person 2 (Backend Developer - FastAPI)

Day-1 Mock Implementation:
Provides a fully contract-compliant black-box function `predict_window()` so
Person 2 and Person 3 can develop, test, and containerize the FastAPI service
without waiting for the final PyTorch ResNet-18 model weights.
"""

import time
from typing import Any, Dict, List


def predict_window(kpi_window: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Evaluates a 16-step chronological window of 3 cellular KPIs.

    Args:
        kpi_window: List of 16 dicts, each containing:
                    - 'PRBUsageDL' (float)
                    - 'meanThr_DL' (float)
                    - 'meanUE_DL' (float)

    Returns:
        dict:
            - status (str): "ANOMALY_DETECTED" or "NORMAL"
            - anomaly_label (int): 1 or 0
            - confidence (float): Probability score in [0.0, 1.0]
            - inference_time_ms (float): Execution time in milliseconds
    """
    start_time = time.perf_counter()

    if len(kpi_window) != 16:
        raise ValueError(f"Expected a window of exactly 16 time steps, got {len(kpi_window)}")

    # Basic heuristic for mock testing before Day 9 model swap-in:
    # If PRB usage > 80% or throughput < 5 Mbps in the latest step, mock anomaly
    last_step = kpi_window[-1]
    is_anomaly = (
        last_step.get("PRBUsageDL", 0) > 80.0
        or (last_step.get("meanThr_DL", 50.0) < 5.0 and last_step.get("meanUE_DL", 0) > 20.0)
    )

    # Simulate realistic CPU inference latency (~15ms)
    time.sleep(0.015)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if is_anomaly:
        return {
            "status": "ANOMALY_DETECTED",
            "anomaly_label": 1,
            "confidence": 0.924,
            "inference_time_ms": round(elapsed_ms, 2),
        }
    else:
        return {
            "status": "NORMAL",
            "anomaly_label": 0,
            "confidence": 0.887,
            "inference_time_ms": round(elapsed_ms, 2),
        }
