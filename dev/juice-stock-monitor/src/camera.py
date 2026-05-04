"""
camera.py - Webカメラ撮影モジュール
"""
import cv2
from pathlib import Path


def capture_image(
    output_path: str,
    device: int = 0,
) -> bool:
    """
    Webカメラで1枚撮影して保存する。

    Args:
        output_path: 保存先ファイルパス
        device: カメラデバイス番号（/dev/video0 → 0）

    Returns:
        成功時True、失敗時False
    """
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"[ERROR] カメラを開けませんでした: /dev/video{device}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] フレームの取得に失敗しました: /dev/video{device}")
        return False

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)

    h, w = frame.shape[:2]
    print(f"[OK] 撮影成功: {output} ({w}x{h})")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / "data" / "results" / "camera_test.jpg"

    print(f"カメラデバイス: /dev/video0")
    print(f"保存先: {output_path}")
    print("-" * 40)

    success = capture_image(str(output_path), device=0)

    if not success:
        print("[FAIL] 撮影に失敗しました")
        raise SystemExit(1)
