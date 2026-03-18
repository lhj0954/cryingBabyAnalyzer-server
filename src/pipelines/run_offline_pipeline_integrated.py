import os
import sys  # sys 모듈을 꼭 import 해야 합니다.

# 🚨 이 부분이 반드시 from utils... 보다 먼저 실행되어야 합니다!
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 pipelines 폴더
src_dir = os.path.abspath(os.path.join(current_dir, "..")) # 상위 폴더 (src)
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
import numpy as np

from utils.audio_io import load_audio
from stages.yamnet_stage import YamnetStage
from stages.ast_stage import ASTStage

TARGET_SR = 16000
HOP_SECONDS = 0.48
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

# YAMNet이 울음을 확실히 감지하기 위한 짧은 모니터링 버퍼 (예: 2초)
YAMNET_BUFFER_SAMPLES = int(TARGET_SR * 2.0)

# AST 모델이 추론하기 가장 좋은 오디오 길이 (예: 5초)
AST_OPTIMAL_SAMPLES = int(TARGET_SR * 5.0)

def main():
    # 1. 파일 경로 설정 (절대 경로)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    input_path = os.path.join(project_root, "samples", "mixed_30s.wav")    
    print("[시스템] 모델과 오디오 파일을 로딩합니다...")
    wav_full, sr = load_audio(input_path)
    
    yamnet = YamnetStage(trigger_threshold=0.20)
    ast_model = ASTStage(model_dir="ast-baby-cry-final")
    
    total_samples = len(wav_full)
    
    print(f"\n=== [실전 제품 모드] 아기 울음 원인 분석 시작 ===")
    print("목표: 울음 감지 즉시 가장 선명한 5초 구간만 잘라 1회 분석 (5초 이내 응답)")
    print("-" * 60)
    
    # 2. YAMNet으로 가볍게 상시 모니터링 (0.48초 단위 슬라이딩)
    for end_idx in range(HOP_SAMPLES, total_samples, HOP_SAMPLES):
        start_idx = max(0, end_idx - YAMNET_BUFFER_SAMPLES)
        
        # YAMNet에 넣을 짧은(최대 2초) 오디오 조각
        yamnet_chunk = wav_full[start_idx:end_idx]
        
        yamnet_result = yamnet.run(yamnet_chunk)
        
        # 3. 처음으로 울음소리가 감지된 순간 (Trigger!)
        if yamnet_result["triggered"]:
            trigger_time_sec = end_idx / TARGET_SR
            print(f"\n[Trigger] 🚨 {trigger_time_sec:.2f}초 지점에서 아기 울음소리 최초 감지!")
            
            # 4. AST 모델에 넣을 '최적의 5초' 구간 자르기
            # 울음 시작점의 소실을 막기 위해 감지 시점보다 1초 앞당겨서 시작점을 잡고 총 5초를 가져옵니다.
            # (※ 실제 스마트폰이라면 이 시점부터 4초간 추가 마이크 녹음을 한 뒤 분석하게 됩니다.)
            ast_start_idx = max(0, end_idx - int(TARGET_SR * 1.0))
            ast_end_idx = min(total_samples, ast_start_idx + AST_OPTIMAL_SAMPLES)
            
            ast_chunk = wav_full[ast_start_idx:ast_end_idx]
            ast_start_sec = ast_start_idx / TARGET_SR
            ast_end_sec = ast_end_idx / TARGET_SR
            
            print(f"   -> ✂️ 오디오 자르기: {ast_start_sec:.2f}초 ~ {ast_end_sec:.2f}초 (길이: {len(ast_chunk)/TARGET_SR:.2f}초)")
            print("   -> 🧠 AST 모델 원인 분석 중...")
            
            # 5. 무거운 AST 모델로 단 1회 집중 추론
            cause, confidence = ast_model.predict(ast_chunk, TARGET_SR)
            
            print(f"   => 💡 최종 분석 결과: {cause} (확률: {confidence*100:.1f}%)")
            
            # 6. 첫 번째 울음에 대한 분석과 알림이 끝났으므로 즉시 루프 종료 (One-Shot)
            break
            
    print("-" * 60)
    print("=== 프로세스 종료 ===")

if __name__ == "__main__":
    main()