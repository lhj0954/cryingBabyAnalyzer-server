import os
import sys
from typing import Optional, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))    # src/server
src_dir = os.path.abspath(os.path.join(current_dir, ".."))  # src

if src_dir not in sys.path:
    sys.path.append(src_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pipelines.infer_api_pipeline import InferApiPipeline

app = FastAPI(title="cryingBabyAnalyzer API")
pipeline = InferApiPipeline()


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
    message: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_info": pipeline.health(),
    }


async def _run_infer(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="현재는 wav만 지원합니다.")

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        return pipeline.predict_from_bytes(file_bytes, filename=file.filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")


@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    return await _run_infer(file)
