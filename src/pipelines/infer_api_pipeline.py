import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))      # src/pipelines
src_dir = os.path.abspath(os.path.join(current_dir, ".."))    # src
project_root = os.path.abspath(os.path.join(src_dir, ".."))   # repo root

if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.audio_io import load_audio_from_bytes
from stages.ast_stage import ASTStage

AST_OPTIMAL_SECONDS = 5.0
MODEL_DIR_NAME = "ast-baby-cry-final"


class InferApiPipeline:
    def __init__(self):
        model_dir = os.path.join(project_root, MODEL_DIR_NAME)
        self.ast_model = ASTStage(model_dir=model_dir)

    def health(self):
        return {
            "model_dir": os.path.join(project_root, MODEL_DIR_NAME),
            "mode": "server_ast_only",
            "labels": ["awake", "diaper", "hug", "hungry", "sleepy", "uncomfortable"],
        }

    def _select_ast_chunk(self, wav_full, sr: int):
        total_samples = len(wav_full)
        ast_optimal_samples = int(sr * AST_OPTIMAL_SECONDS)

        # 앱에서 이미 트리거된 5초 클립을 보내는 구조라서
        # 서버는 앞부분부터 최대 5초만 분석
        start_idx = 0
        end_idx = min(total_samples, ast_optimal_samples)
        ast_chunk = wav_full[start_idx:end_idx]

        return ast_chunk, start_idx, end_idx, total_samples

    def predict_from_bytes(self, file_bytes: bytes, filename: str = "unknown.wav"):
        wav_full, sr = load_audio_from_bytes(file_bytes)

        if wav_full is None or len(wav_full) == 0:
            return {
                "filename": filename,
                "sample_rate": sr if sr else 0,
                "duration_sec": 0.0,
                "triggered": False,
                "trigger_time_sec": None,
                "analysis_window": None,
                "prediction": None,
                "message": "오디오가 비어 있습니다."
            }

        ast_chunk, ast_start_idx, ast_end_idx, total_samples = self._select_ast_chunk(wav_full, sr)

        pred_result = self.ast_model.predict(ast_chunk, sr)

        # ASTStage.predict()가 현재 (label, confidence)만 반환하는 구조 기준
        scores = None
        if isinstance(pred_result, tuple):
            if len(pred_result) == 2:
                cause, confidence = pred_result
            elif len(pred_result) == 3:
                cause, confidence, scores = pred_result
            else:
                raise ValueError("ASTStage.predict() 반환 형식이 올바르지 않습니다.")
        elif isinstance(pred_result, dict):
            cause = pred_result["label"]
            confidence = pred_result["confidence"]
            scores = pred_result.get("scores")
        else:
            raise ValueError("ASTStage.predict() 반환 형식을 해석할 수 없습니다.")

        prediction = {
            "label": cause,
            "confidence": round(float(confidence), 4),
            "yamnet": {
                "baby_cry_max": 0.0,
                "crying_max": 0.0,
                "merged_cry_max": 0.0,
            }
        }

        # 나중에 main.py의 Prediction 모델에 scores 필드를 추가하면 같이 내려보낼 수 있음
        if scores is not None:
            prediction["scores"] = scores

        result = {
            "filename": filename,
            "sample_rate": sr,
            "duration_sec": round(total_samples / sr, 3),
            "triggered": True,
            "trigger_time_sec": 0.0,
            "analysis_window": {
                "start_sec": round(ast_start_idx / sr, 3),
                "end_sec": round(ast_end_idx / sr, 3),
                "duration_sec": round(len(ast_chunk) / sr, 3),
            },
            "prediction": prediction,
            "message": "추론 완료"
        }

        return result