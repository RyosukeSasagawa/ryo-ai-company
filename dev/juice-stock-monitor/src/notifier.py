"""
notifier.py - LINE通知モジュール
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")
LINE_API_URL = "https://api.line.me/v2/bot/message/push"


def send_line_message(message: str) -> bool:
    """
    LINE Messaging APIでメッセージを送信する。

    Args:
        message: 送信するテキスト

    Returns:
        成功時True、失敗時False
    """
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("[ERROR] LINE_CHANNEL_ACCESS_TOKEN または LINE_USER_ID が未設定です")
        return False

    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message}],
    }

    response = requests.post(LINE_API_URL, headers=headers, json=payload, timeout=10)

    if response.status_code == 200:
        print(f"[LINE] 送信成功: {message}")
        return True
    else:
        print(f"[ERROR] 送信失敗: status={response.status_code}, body={response.text}")
        return False


def notify_if_needed(judge_result: dict) -> bool:
    """
    judge_stock()の結果を受け取り、needs_alertがTrueの時だけLINE通知する。

    Args:
        judge_result: judge_stock()が返す辞書
            {"status": str, "bottle_count": int, "needs_alert": bool}

    Returns:
        通知を送った場合True、スキップした場合False
    """
    if not judge_result.get("needs_alert"):
        print(f"[通知スキップ] ステータス: {judge_result['status']} (在庫十分)")
        return False

    status = judge_result["status"]
    count = judge_result["bottle_count"]

    if status == "EMPTY":
        message = f"【在庫警告】ジュースが0本です！今すぐ補充してください。"
    else:
        message = f"【在庫警告】ジュースの残りが少なくなっています（残り{count}本）。補充をご検討ください。"

    return send_line_message(message)


if __name__ == "__main__":
    from pathlib import Path
    from ultralytics import YOLO
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from detector import detect_objects
    from stock_judge import judge_stock

    base_dir = Path(__file__).parent.parent
    image_path = base_dir / "data" / "sample_images" / "low.png"
    model_path = base_dir / "yolov8n.pt"

    print(f"テスト画像: {image_path.name}")
    model = YOLO(str(model_path))
    model.to("cpu")

    detections = detect_objects(str(image_path), model)
    result = judge_stock(detections)

    print(f"bottle数      : {result['bottle_count']} 本")
    print(f"在庫ステータス: {result['status']}")
    print(f"通知要否      : {'要' if result['needs_alert'] else '不要'}")
    print("-" * 40)

    notified = notify_if_needed(result)
    print(f"通知結果: {'送信済み' if notified else 'スキップ'}")
