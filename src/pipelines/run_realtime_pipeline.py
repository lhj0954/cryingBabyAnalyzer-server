#실시간 파이프 라인
from collections import deque
import numpy as np
import sounddevice as sd

from yamnet_stage import YamnetStage

TARGET_SR = 16000
CHANNELS = 1

# 0.48초마다 추론
HOP_SECONDS = 0.48
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

# 최근 2초 버퍼 유지
BUFFER_SECONDS = 2.0
BUFFER_SAMPLES = int(TARGET_SR * BUFFER_SECONDS)


def main():
    yamnet = YamnetStage(trigger_threshold=0.20)

    audio_buffer = deque(maxlen=BUFFER_SAMPLES)

    print("=== YAMNET REALTIME TEST ===")
    print("마이크 입력 시작. 종료는 Ctrl+C")

    with sd.InputStream(
        samplerate=TARGET_SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=HOP_SAMPLES,
    ) as stream:
        try:
            while True:
                indata, overflowed = stream.read(HOP_SAMPLES)
                if overflowed:
                    print("경고: overflow 발생")

                chunk = indata[:, 0]
                for x in chunk:
                    audio_buffer.append(float(x))

                if len(audio_buffer) < TARGET_SR:
                    # 최소 1초는 쌓인 뒤부터 보기
                    continue

                wav = np.array(audio_buffer, dtype=np.float32)
                wav = np.clip(wav, -1.0, 1.0)

                result = yamnet.run(wav)

                top_last = result["top_classes_per_frame"][-1]
                baby_last = result["baby_cry_scores"][-1]
                crying_last = result["crying_scores"][-1]
                merged_last = result["merged_cry_scores"][-1]

                print(
                    f"top={top_last} | "
                    f"baby_cry={baby_last:.4f} | "
                    f"crying={crying_last:.4f} | "
                    f"merged={merged_last:.4f} | "
                    f"triggered={result['triggered']}"
                )

                if result["triggered"]:
                    print(">>> 아기 울음 트리거 감지")

        except KeyboardInterrupt:
            print("\n종료됨")


if __name__ == "__main__":
    main()