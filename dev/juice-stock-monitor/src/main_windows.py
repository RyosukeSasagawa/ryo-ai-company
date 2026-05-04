"""
main_windows.py - ジュース在庫監視メインスクリプト（Windows版）
カメラ撮影→YOLOv8検出→在庫判定→LINE通知の一連の流れを実行する
実行環境：Windows 11 Pro / Python 3.10
"""
import cv2
import os
import requests
import time
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

def capture():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("capture.jpg", frame)
        return "capture.jpg"
    return None

def detect(image_path):
    model = YOLO("yolov8n.pt")
    results = model(image_path, conf=0.25, verbose=False, device="cpu")
    count = sum(1 for r in results for b in r.boxes if r.names[int(b.cls)] == "bottle")
    return count

def judge(count):
    if count >= 10:
        return "FULL", False
    elif count >= 1:
        return "LOW", True
    else:
        return "EMPTY", True

def notify(status, count):
    if status == "EMPTY":
        msg = "【在庫警告】ジュースが0本です！今すぐ補充してください。"
    else:
        msg = f"【在庫警告】ジュースの残りが少なくなっています（残り{count}本）。"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": msg}]}
    requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload)
    print(f"[LINE通知] {msg}")

if __name__ == "__main__":
    print("撮影中...")
    image = capture()
    if not image:
        print("[ERROR] 撮影失敗")
        exit()
    print("検出中...")
    count = detect(image)
    status, needs_alert = judge(count)
    print(f"bottle数: {count}本 / ステータス: {status}")
    if needs_alert:
        notify(status, count)
    else:
        print("在庫十分。通知不要。")
