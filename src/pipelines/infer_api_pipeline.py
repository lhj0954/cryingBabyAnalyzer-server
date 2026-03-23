import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))      # src/pipelines
src_dir = os.path.abspath(os.path.join(current_dir, ".."))    # src
project_root = os.path.abspath(os.path.join(src_dir, ".."))   # repo root

if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.audio_io import load_audio_from_bytes
from stages.yamnet_stage import YamnetStage
from stages.ast_stage import ASTStage

TARGET_SR = 16000
HOP_SECONDS = 0.48
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

YAMNET_BUFFER_SAMPLES = int(TARGET_SR * 2.0)
AST_OPTIMAL_SAMPLES = int(TARGET_SR * 5.0)

# 팀원한테 받은 실제 모델 폴더명으로 바꿔
MODEL_DIR_NAME = "ast-baby-cry-final"


class InferApiPipeline:
    def __init__(self):
        model_dir = os.path.join(project_root, MODEL_DIR_NAME)

        self.yamnet = YamnetStage(trigger_threshold=0.20)
        self.ast_model = ASTStage(model_dir=model_dir)

    def predict_from_bytes(self, file_bytes: bytes, filename: str = "unknown.wav"):
        wav_full, sr = load_audio_from_bytes(file_bytes)
        total_samples = len(wav_full)

        result = {
            "filename": filename,
            "sample_rate": sr,
            "duration_sec": round(total_samples / sr, 3),
            "triggered": False,
            "trigger_time_sec": None,
            "analysis_window": None,
            "prediction": None,
            "message": "울음 트리거 안 됨"
        }

        for end_idx in range(HOP_SAMPLES, total_samples + 1, HOP_SAMPLES):
            start_idx = max(0, end_idx - YAMNET_BUFFER_SAMPLES)
            yamnet_chunk = wav_full[start_idx:end_idx]
            yamnet_result = self.yamnet.run(yamnet_chunk)

            if yamnet_result["triggered"]:
                trigger_time_sec = end_idx / TARGET_SR

                ast_start_idx = max(0, end_idx - int(TARGET_SR * 1.0))
                ast_end_idx = min(total_samples, ast_start_idx + AST_OPTIMAL_SAMPLES)
                ast_chunk = wav_full[ast_start_idx:ast_end_idx]

                cause, confidence = self.ast_model.predict(ast_chunk, TARGET_SR)

                result["triggered"] = True
                result["trigger_time_sec"] = round(trigger_time_sec, 3)
                result["analysis_window"] = {
                    "start_sec": round(ast_start_idx / TARGET_SR, 3),
                    "end_sec": round(ast_end_idx / TARGET_SR, 3),
                    "duration_sec": round(len(ast_chunk) / TARGET_SR, 3),
                }
                result["prediction"] = {
                    "label": cause,
                    "confidence": round(float(confidence), 4),
                    "yamnet": {
                        "baby_cry_max": round(float(yamnet_result["baby_cry_max"]), 4),
                        "crying_max": round(float(yamnet_result["crying_max"]), 4),
                        "merged_cry_max": round(float(yamnet_result["merged_cry_max"]), 4),
                    }
                }
                result["message"] = "추론 완료"
                break

        return result