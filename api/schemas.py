"""
Pydantic Schemas for Telecom QoS Anomaly Detection API
Owned by: Person 2 (Backend Developer)
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class KPIReading(BaseModel):
    PRBUsageDL: float = Field(..., ge=0.0, le=100.0, description="Radio Resource Block usage % (0-100)")
    meanThr_DL: float = Field(..., ge=0.0, description="Mean downlink throughput (Mbps)")
    meanUE_DL: float = Field(..., ge=0.0, description="Mean active User Equipment count")


class PredictRequest(BaseModel):
    cell_name: str = Field(..., example="3BLTE", description="Base station identifier / sector")
    window: List[KPIReading] = Field(..., description="Chronological sliding window of 16 consecutive 15-min KPI readings")

    @field_validator("window")
    def validate_window_length(cls, v):
        if len(v) != 16:
            raise ValueError(f"Window must contain exactly 16 chronological readings, received {len(v)}")
        return v


class PredictResponse(BaseModel):
    cell_name: str
    status: str = Field(..., example="ANOMALY_DETECTED")
    anomaly_label: int = Field(..., example=1)
    confidence: float = Field(..., example=0.924)
    inference_time_ms: float = Field(..., example=15.2)


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    service: str = "Telecom QoS Anomaly Detection API"
