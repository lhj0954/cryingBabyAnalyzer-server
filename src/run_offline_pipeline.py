# 파이프 라인 실행
# 결과 출력
from audio_io import load_audio
from yamnet_stage import YamnetStage


def main():
    input_path = "samples/mixed_30s.wav"

    wav, sr = load_audio(input_path)

    print(f"sample_rate: {sr}")
    print(f"num_samples: {len(wav)}")
    print(f"duration_sec: {len(wav)/sr:.3f}")
    print(f"max_abs: {abs(wav).max():.6f}")
    print(f"mean_abs: {abs(wav).mean():.6f}")

    yamnet = YamnetStage(trigger_threshold=0.20)
    result = yamnet.run(wav)

    print("=== YAMNET OFFLINE RESULT ===")
    print(f"file: {input_path}")
    print(f"frame_count: {len(result['top_classes_per_frame'])}")
    print(f"baby_cry_max: {result['baby_cry_max']:.4f}")
    print(f"crying_max: {result['crying_max']:.4f}")
    print(f"merged_cry_max: {result['merged_cry_max']:.4f}")
    print(f"triggered: {result['triggered']}")

    print("\n[Frame Top Classes]")
    for i, cls_name in enumerate(result["top_classes_per_frame"], start=1):
        baby_score = result["baby_cry_scores"][i - 1]
        crying_score = result["crying_scores"][i - 1]
        merged_score = result["merged_cry_scores"][i - 1]

        print(
            f"[{i}] top={cls_name} | "
            f"baby_cry={baby_score:.4f} | "
            f"crying={crying_score:.4f} | "
            f"merged={merged_score:.4f}"
        )

    if result["triggered"]:
        print("\n최종 판단: 아기 울음 후보로 트리거됨")
    else:
        print("\n최종 판단: 아기 울음 트리거 안 됨")


if __name__ == "__main__":
    main()