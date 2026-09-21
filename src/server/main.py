import os
import sys
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
import uuid

current_dir = os.path.dirname(os.path.abspath(__file__))    # src/server
src_dir = os.path.abspath(os.path.join(current_dir, ".."))  # src

if current_dir not in sys.path:
    sys.path.append(current_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pipelines.infer_api_pipeline import InferApiPipeline
from records_db import init_db, insert_record, list_records, get_record, get_stats


app = FastAPI(title="cryingBabyAnalyzer API")
pipeline = InferApiPipeline()

UPLOAD_DIR = Path("uploaded_audio")
UPLOAD_DIR.mkdir(exist_ok=True)
init_db()


class AnalysisWindow(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float


class YamnetInfo(BaseModel):
    baby_cry_max: float = 0.0
    crying_max: float = 0.0
    merged_cry_max: float = 0.0


class Prediction(BaseModel):
    label: str
    confidence: float
    scores: Optional[Dict[str, float]] = None
    yamnet: YamnetInfo


class InferResponse(BaseModel):
    filename: str
    sample_rate: int
    duration_sec: float
    triggered: bool
    trigger_time_sec: Optional[float] = None
    analysis_window: Optional[AnalysisWindow] = None
    prediction: Optional[Prediction] = None
    record_id: Optional[int] = None
    message: str


class CryRecord(BaseModel):
    id: int
    created_at: str
    filename: str
    saved_filename: str
    audio_path: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    duration_sec: Optional[float] = None
    message: Optional[str] = None


class LabelCount(BaseModel):
    label: Optional[str] = None
    count: int


class HourCount(BaseModel):
    hour: str
    count: int


class RecordsStats(BaseModel):
    total: int
    by_label: List[LabelCount]
    by_hour: List[HourCount]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/records", response_model=List[CryRecord])
def records(date: Optional[str] = None, label: Optional[str] = None, limit: int = 100):
    return list_records(date=date, label=label, limit=limit)


@app.get("/records/stats", response_model=RecordsStats)
def records_stats(date: Optional[str] = None):
    return get_stats(date=date)


@app.get("/records/{record_id}", response_model=CryRecord)
def record_detail(record_id: int):
    record = get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return record


async def _run_infer(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="현재는 wav만 지원합니다.")

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
        saved_path = UPLOAD_DIR / saved_name

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        result = pipeline.predict_from_bytes(file_bytes, filename=file.filename)

        prediction = result.get("prediction") if isinstance(result, dict) else None
        label = None
        confidence = None

        if prediction:
            label = prediction.get("label")
            confidence = prediction.get("confidence")

        record_id = insert_record(
            created_at=created_at,
            filename=file.filename,
            saved_filename=saved_name,
            audio_path=str(saved_path),
            label=label,
            confidence=confidence,
            duration_sec=result.get("duration_sec") if isinstance(result, dict) else None,
            message=result.get("message") if isinstance(result, dict) else None,
        )

        if isinstance(result, dict):
            result["record_id"] = record_id

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")


@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    return await _run_infer(file)


@app.post("/predict", response_model=InferResponse)
async def predict(file: UploadFile = File(...)):
    return await _run_infer(file)
