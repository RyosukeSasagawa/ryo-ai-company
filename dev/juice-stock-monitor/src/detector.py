"""
detector.py - YOLOv8による物体検出モジュール
"""
from pathlib import Path
from ultralytics import YOLO


def detect_objects(image_path: str, model: YOLO, conf_threshold: float = 0.25) -> list[dict]:
    """
    画像から物体を検出して結果を返す。

    Args:
        image_path: 画像ファイルのパス
        model: ロード済みYOLOモデル
        conf_threshold: 信頼度の閾値（デフォルト0.25）

    Returns:
        検出結果のリスト。各要素は {"name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    results = model(image_path, conf=conf_threshold, verbose=False, device="cpu")
    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "name": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            })

    return detections


def process_sample_images(image_dir: str, model_name: str = "yolov8n.pt") -> None:
    """
    指定ディレクトリのpng画像を順番に処理して結果を表示する。

    Args:
        image_dir: 画像ディレクトリのパス
        model_name: 使用するYOLOモデル名
    """
    model = YOLO(model_name)
    model.to("cpu")
    image_paths = sorted(Path(image_dir).glob("*.png"))

    if not image_paths:
        print(f"[WARNING] PNG画像が見つかりません: {image_dir}")
        return

    print(f"モデル: {model_name}")
    print(f"画像数: {len(image_paths)} 枚")
    print("=" * 60)

    for image_path in image_paths:
        print(f"\n[画像] {image_path.name}")
        detections = detect_objects(str(image_path), model)

        if not detections:
            print("  検出なし")
            continue

        for i, det in enumerate(detections, 1):
            print(f"  [{i}] {det['name']}")
            print(f"       信頼度: {det['confidence']:.2%}")
            print(f"       座標  : {det['bbox']}")

    print("\n" + "=" * 60)
    print("処理完了")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    image_dir = base_dir / "data" / "sample_images"
    process_sample_images(str(image_dir))
