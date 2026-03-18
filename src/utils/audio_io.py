# wav파일 읽기
# mono 변환
# sample rate 맞추기
# numpy/torch 형태 정리
import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16000


def load_audio(path: str):
    wav, sr = sf.read(path, dtype="float32")

    # stereo -> mono
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    # resample to 16k if needed
    if sr != TARGET_SR:
        target_length = int(round(len(wav) / sr * TARGET_SR))
        wav = signal.resample(wav, target_length).astype("float32")
        sr = TARGET_SR

    # clamp just in case
    wav = np.clip(wav, -1.0, 1.0).astype("float32")
    return wav, sr


def save_audio(path: str, audio: np.ndarray, sr: int = TARGET_SR):
    sf.write(path, audio, sr)

    