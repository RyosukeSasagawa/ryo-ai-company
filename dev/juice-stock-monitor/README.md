# juice-stock-monitor
ミニPC+WebカメラによるジュースストックAI監視システム。

## 概要
エッジPC（ASUS PN42 / Intel N100）にWebカメラを接続し、
YOLOv8で在庫を検出。在庫不足時にLINEへ自動通知する。

## システム構成
Webカメラ → エッジPC（YOLOv8推論）→ 在庫判定 → LINE通知
↓
AWS S3（画像保存）※実装予定

## 技術スタック
- エッジ推論：YOLOv8n（ultralytics）/ CPU推論
- 画像処理：OpenCV / Pillow
- 通知：LINE Messaging API
- クラウド：AWS S3（実装予定）
- 言語：Python 3.12
- 実行環境：Windows 11 Pro（エッジPC想定）/ Python 3.10

## 在庫判定ロジック
| 状態 | bottle数 | 通知 |
|------|---------|------|
| FULL | 10本以上 | なし |
| LOW  | 1〜9本  | LINE通知 |
| EMPTY| 0本     | LINE通知 |

## プロジェクト構成
juice-stock-monitor/
├── src/
│   ├── detector.py      # YOLOv8物体検出
│   ├── stock_judge.py   # 在庫判定ロジック
│   └── notifier.py      # LINE通知
├── data/
│   ├── sample_images/   # テスト用画像
│   └── results/         # 検出結果画像
├── requirements.txt
└── .env.example

## 実装状況
- [x] Phase1：プロジェクト基盤構築
- [x] Phase2：YOLOv8物体検出
- [x] Phase3：在庫判定ロジック
- [x] Phase4：LINE通知
- [ ] Phase5：AWS S3連携
- [x] Phase6：Webカメラ動作確認（Windows環境）
- [x] Phase7：カメラ+AI統合・LINE通知（Windows環境で実機動作確認済み）
- [ ] Phase8：定期実行・自動化

## セットアップ
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .envにLINEトークン等を設定
```
