"""
s3_uploader.py - AWS S3画像アップロードモジュール
撮影した画像をS3バケットに保存する
実行環境：Windows 11 Pro / Python 3.10
"""
import boto3
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def upload_to_s3(image_path: str, status: str) -> str:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"captures/{status}/{timestamp}.jpg"
    s3.upload_file(image_path, AWS_BUCKET_NAME, key)
    print(f"[S3] アップロード完了: s3://{AWS_BUCKET_NAME}/{key}")
    return key


if __name__ == "__main__":
    if os.path.exists("capture.jpg"):
        upload_to_s3("capture.jpg", "TEST")
    else:
        print("[ERROR] capture.jpgが見つかりません。先にmain.pyを実行してください。")
