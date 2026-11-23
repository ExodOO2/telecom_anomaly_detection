"""
Pydantic Schemas for Telecom QoS Anomaly Detection API
Owned by: Person 2 (Backend Developer & API Engineer)

These schemas provide strict data validation, boundary checks, and self-documenting
OpenAPI/Swagger UI schemas for the production REST service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class KPIReading(BaseModel):
    """
    Telemetry reading representing one 15-minute interval for a single antenna sector.
    Enforces physical telecom domain boundaries.
    """
    PRBUsageDL: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Downlink Physical Resource Block radio congestion % (0.0 - 100.0%)",
        examples=[18.5]
    )
    meanThr_DL: float = Field(
        ...,
        ge=0.0,
        description="Delivered downlink throughput speed in Mbps (non-negative)",
        examples=[4.25]
    )
    meanUE_DL: float = Field(
        ...,
        ge=0.0,
        description="Average active User Equipment (connected mobile devices) count",
        examples=[1.85]
    )


class PredictRequest(BaseModel):
    """
    Inference request payload containing cell metadata, operational threshold,
    and a strict 16-step chronological telemetry window (4 hours of data).
    """
    cell_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Cell tower / sector identifier (e.g., 3BLTE, 10ALTE)",
        examples=["3BLTE"]
    )
    threshold: Optional[float] = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description=(
            "Decision threshold for anomaly classification. "
            "0.50 = Safety Net mode (70.7% recall); "
            "0.66 = Balanced mode (optimal F1, reduces false alarms by 45%)."
        ),
        examples=[0.50]
    )
    window: List[KPIReading] = Field(
        ...,
        description="Chronological sliding window of exactly 16 consecutive 15-minute KPI readings (4 hours total)."
    )

    @field_validator("window")
    def validate_exact_window_length(cls, v: List[KPIReading]) -> List[KPIReading]:
        if len(v) != 16:
            raise ValueError(
                f"Sliding window must contain exactly 16 chronological readings (4 hours). "
                f"Received {len(v)} readings."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "cell_name": "3BLTE",
                "threshold": 0.50,
                "window": [
                    {"PRBUsageDL": 3.133, "meanThr_DL": 0.875, "meanUE_DL": 1.172},
                    {"PRBUsageDL": 1.213, "meanThr_DL": 0.324, "meanUE_DL": 1.132},
                    {"PRBUsageDL": 0.606, "meanThr_DL": 0.102, "meanUE_DL": 1.031},
                    {"PRBUsageDL": 0.707, "meanThr_DL": 0.066, "meanUE_DL": 1.061},
                    {"PRBUsageDL": 1.584, "meanThr_DL": 0.562, "meanUE_DL": 1.122},
                    {"PRBUsageDL": 4.345, "meanThr_DL": 1.688, "meanUE_DL": 1.091},
                    {"PRBUsageDL": 0.909, "meanThr_DL": 0.273, "meanUE_DL": 1.101},
                    {"PRBUsageDL": 0.707, "meanThr_DL": 0.120, "meanUE_DL": 1.051},
                    {"PRBUsageDL": 1.111, "meanThr_DL": 0.354, "meanUE_DL": 1.081},
                    {"PRBUsageDL": 1.818, "meanThr_DL": 0.765, "meanUE_DL": 1.152},
                    {"PRBUsageDL": 2.525, "meanThr_DL": 1.020, "meanUE_DL": 1.212},
                    {"PRBUsageDL": 3.232, "meanThr_DL": 1.340, "meanUE_DL": 1.283},
                    {"PRBUsageDL": 2.929, "meanThr_DL": 1.110, "meanUE_DL": 1.242},
                    {"PRBUsageDL": 2.020, "meanThr_DL": 0.890, "meanUE_DL": 1.182},
                    {"PRBUsageDL": 1.414, "meanThr_DL": 0.540, "meanUE_DL": 1.111},
                    {"PRBUsageDL": 1.010, "meanThr_DL": 0.410, "meanUE_DL": 1.071}
                ]
            }
        }
    }


class PredictResponse(BaseModel):
    """
    Anomaly classification response returned by the ResNet-18 Computer Vision model.
    """
    cell_name: str = Field(..., description="Base station identifier")
    status: str = Field(..., description="'NORMAL' or 'ANOMALY_DETECTED'", examples=["NORMAL"])
    anomaly_label: int = Field(..., description="0 for Normal, 1 for Anomaly", examples=[0])
    confidence: float = Field(..., description="Probability confidence of the predicted state (0.0 - 1.0)", examples=[0.887])
    threshold_applied: float = Field(..., description="The decision threshold used for classification", examples=[0.50])
    inference_time_ms: float = Field(..., description="Raw model execution latency in milliseconds", examples=[12.45])


class BatchPredictRequest(BaseModel):
    """
    Batch prediction request allowing multi-cell evaluation in a single API call.
    """
    requests: List[PredictRequest] = Field(..., description="List of cell sliding window requests")


class BatchPredictResponse(BaseModel):
    """
    Batch prediction response summarizing multiple cell evaluations.
    """
    total_cells: int = Field(..., description="Total number of evaluated base stations")
    anomalies_detected: int = Field(..., description="Count of base stations flagged with anomalies")
    results: List[PredictResponse] = Field(..., description="Individual cell prediction outcomes")
    total_batch_time_ms: float = Field(..., description="Total execution time for the batch in milliseconds")


class HealthResponse(BaseModel):
    """
    Health check payload confirming service status and loaded model artifacts.
    """
    status: str = Field(default="healthy", description="API operational health status")
    model_loaded: bool = Field(default=True, description="True if ResNet-18 weights and scaler are active in memory")
    model_architecture: str = Field(default="ResNet-18 (GASF 3-Channel RGB)", description="Model backbone")
    service: str = Field(default="Telecom QoS Anomaly Detection API", description="Service title")
    version: str = Field(default="1.0.0", description="API release version")
