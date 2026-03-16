#Silero VAD 로드
#speech timesatmps 추출
#후보 구간 잘라내기
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class YamnetStage:
    def __init__(self, trigger_threshold: float = 0.20):
        self.trigger_threshold = trigger_threshold
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_names = self._load_class_names()

        # cry 관련 클래스 인덱스 찾기
        self.baby_cry_idx = self._find_class_index("Baby cry, infant cry")
        self.crying_idx = self._find_class_index("Crying, sobbing")

    def _load_class_names(self):
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        class_names = []

        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row["display_name"])

        return class_names

    def _find_class_index(self, target_name: str):
        for i, name in enumerate(self.class_names):
            if name == target_name:
                return i
        raise ValueError(f"클래스를 찾지 못했어: {target_name}")

    def run(self, wav_np: np.ndarray):
        """
        wav_np: mono 16kHz float32 numpy array in [-1, 1]
        return:
          {
            "scores": ...,
            "embeddings": ...,
            "spectrogram": ...,
            "top_classes_per_frame": ...,
            "baby_cry_max": ...,
            "crying_max": ...,
            "triggered": bool
          }
        """
        waveform = tf.convert_to_tensor(wav_np, dtype=tf.float32)

        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()
        embeddings_np = embeddings.numpy()
        spectrogram_np = spectrogram.numpy()

        # 각 프레임의 top1 클래스
        top_class_indices = np.argmax(scores_np, axis=1)
        top_classes_per_frame = [self.class_names[i] for i in top_class_indices]

        baby_cry_scores = scores_np[:, self.baby_cry_idx]
        crying_scores = scores_np[:, self.crying_idx]
        merged_cry_scores = np.maximum(baby_cry_scores, crying_scores)

        baby_cry_max = float(np.max(baby_cry_scores)) if len(baby_cry_scores) else 0.0
        crying_max = float(np.max(crying_scores)) if len(crying_scores) else 0.0
        merged_cry_max = float(np.max(merged_cry_scores)) if len(merged_cry_scores) else 0.0

        # 단순 트리거 규칙:
        # 1) cry 관련 최고 점수가 threshold 이상이거나
        # 2) 연속 두 프레임 이상 0.10 이상이면 트리거
        consecutive_hits = 0
        consecutive_trigger = False
        for s in merged_cry_scores:
            if s >= 0.10:
                consecutive_hits += 1
                if consecutive_hits >= 2:
                    consecutive_trigger = True
                    break
            else:
                consecutive_hits = 0

        triggered = (merged_cry_max >= self.trigger_threshold) or consecutive_trigger

        return {
            "scores": scores_np,
            "embeddings": embeddings_np,
            "spectrogram": spectrogram_np,
            "top_classes_per_frame": top_classes_per_frame,
            "baby_cry_scores": baby_cry_scores,
            "crying_scores": crying_scores,
            "merged_cry_scores": merged_cry_scores,
            "baby_cry_max": baby_cry_max,
            "crying_max": crying_max,
            "merged_cry_max": merged_cry_max,
            "triggered": triggered,
        }