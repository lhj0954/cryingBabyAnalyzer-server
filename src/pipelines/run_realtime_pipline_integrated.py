from collections import deque
import time
import os
import sys  # sys 모듈을 꼭 import 해야 합니다.

# 🚨 이 부분이 반드시 from utils... 보다 먼저 실행되어야 합니다!
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 pipelines 폴더
src_dir = os.path.abspath(os.path.join(current_dir, "..")) # 상위 폴더 (src)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import sounddevice as sd

from stages.yamnet_stage import YamnetStage
from stages.ast_stage import ASTStage  # 새로 만든 AST 클래스 임포트

TARGET_SR = 16000
CHANNELS = 1

HOP_SECONDS = 0.48
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

# 1. YAMNet 상시 모니터링 버퍼 (빠른 반응을 위해 2초만 유지)
YAMNET_BUFFER_SECONDS = 2.0
YAMNET_BUFFER_SAMPLES = int(TARGET_SR * YAMNET_BUFFER_SECONDS)

# 2. AST 분석을 위한 최적의 오디오 길이 (5초)
AST_OPTIMAL_SECONDS = 5.0
AST_OPTIMAL_SAMPLES = int(TARGET_SR * AST_OPTIMAL_SECONDS)

# 3. 알림 발송 후 휴식 시간 (예: 중복 알림 방지 및 배터리 절약을 위한 10초 쿨다운)
COOLDOWN_SECONDS = 10.0

def main():
    print("[시스템] 모델을 로딩합니다...")
    yamnet = YamnetStage(trigger_threshold=0.20)
    ast_model = ASTStage(model_dir="ast-baby-cry-final")
    
    # YAMNet용 2초짜리 롤링 버퍼
    audio_buffer = deque(maxlen=YAMNET_BUFFER_SAMPLES)
    
    # 파이프라인 상태 제어 변수
    is_gathering = False        # 현재 5초 오디오를 수집 중인지 여부
    gathering_buffer = []       # AST에 넣을 5초짜리 오디오를 담는 리스트
    cooldown_until = 0.0        # 쿨다운이 끝나는 시간

    print("\n=== [실전 앱 모드] 아기 울음 실시간 감지 시작 ===")
    print("마이크 입력 대기 중... (종료는 Ctrl+C)")

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
                    print("경고: 오디오 버퍼 오버플로우 발생")

                # numpy 배열을 파이썬 리스트로 변환하여 처리
                chunk = indata[:, 0].tolist()

                # ------------------------------------------------
                # State 1: 쿨다운 상태 (분석 완료 후 휴식 중)
                # ------------------------------------------------
                if time.time() < cooldown_until:
                    # 쿨다운 중에는 과거 소리가 버퍼에 쌓이지 않게 계속 비워줌
                    audio_buffer.clear()
                    continue

                # ------------------------------------------------
                # State 2: 울음 감지 후 5초 오디오 수집 모드 (AST 추론용)
                # ------------------------------------------------
                if is_gathering:
                    gathering_buffer.extend(chunk)
                    
                    # 딱 5초 분량(80,000 샘플)이 채워졌을 때
                    if len(gathering_buffer) >= AST_OPTIMAL_SAMPLES:
                        print(f"   -> ✂️ 5초 길이 오디오 캡처 완료! AST 모델 원인 분석 중...")
                        
                        # 리스트를 numpy 배열로 변환하여 AST 추론
                        ast_wav = np.array(gathering_buffer[:AST_OPTIMAL_SAMPLES], dtype=np.float32)
                        cause, confidence = ast_model.predict(ast_wav, TARGET_SR)
                        
                        print(f"   => 💡 최종 분석 결과: {cause} (확률: {confidence*100:.1f}%)")
                        print("   => 📱 앱 알림 전송 완료! (중복 알림 방지를 위해 10초간 대기합니다)\n")
                        
                        # 분석 완료 후 상태 초기화 및 쿨다운 진입
                        is_gathering = False
                        gathering_buffer = []
                        cooldown_until = time.time() + COOLDOWN_SECONDS
                        audio_buffer.clear()
                        
                    continue

                # ------------------------------------------------
                # State 3: 평상시 모니터링 모드 (YAMNet)
                # ------------------------------------------------
                audio_buffer.extend(chunk)

                # 버퍼에 최소 1초의 데이터가 쌓였을 때만 YAMNet 가동
                if len(audio_buffer) < TARGET_SR:
                    continue

                wav_np = np.array(audio_buffer, dtype=np.float32)
                yamnet_result = yamnet.run(wav_np)

                # 울음소리가 YAMNet의 임계값을 넘었을 때 (Trigger)
                if yamnet_result["triggered"]:
                    print("\n[Trigger] 🚨 아기 울음소리 최초 감지!")
                    print("   -> 가장 깨끗한 분석을 위해 향후 4초간 오디오를 추가 녹음합니다...")
                    
                    is_gathering = True
                    
                    # 💡 핵심: 울음의 "시작 지점"이 잘려나가는 것을 방지하기 위해, 
                    # 이미 YAMNet 버퍼에 들어와 있는 가장 최근 1초 분량을 가져와 AST 버퍼의 앞부분에 채워 넣습니다.
                    past_samples = list(audio_buffer)[-TARGET_SR:]
                    gathering_buffer = past_samples.copy()
                    
                    # YAMNet 모니터링 버퍼 비우기
                    audio_buffer.clear()

        except KeyboardInterrupt:
            print("\n파이프라인이 종료되었습니다.")

if __name__ == "__main__":
    main()