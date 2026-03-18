import torch
import numpy as np
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# 모델 불러오고 실행

class ASTStage:
    def __init__(self, model_dir="ast-baby-cry-final"):
        # 모델과 특징 추출기 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AST] 디바이스 세팅: {self.device}")
        
        # HuggingFace Transformers 기반 AST 모델이라고 가정
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_dir)
        self.model = ASTForAudioClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # 모델이 학습된 라벨 (예시: 배고픔, 아픔, 졸림 등)
        self.id2label = self.model.config.id2label

    def predict(self, wav_np: np.ndarray, sr: int = 16000):
        """
        16kHz mono numpy array를 입력받아 울음의 원인을 예측합니다.
        """
        # 1. 입력 오디오 전처리 (AST 입력 스펙트로그램으로 변환)
        inputs = self.feature_extractor(
            wav_np, 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. 모델 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # 3. 결과 맵핑
        predicted_label = self.id2label[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()

        return predicted_label, confidence