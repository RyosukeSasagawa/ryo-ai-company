"""
Phase 2: OpenCV 色認識
- PyBulletのカメラ画像をキャプチャ
- OpenCV(HSV色空間)で赤・青・黄の球体を検出
- バウンディングボックスと色ラベルを描画
- WSL2対応: cv2.imshow不使用、画像をoutput/に保存
"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
from pathlib import Path

# ---- 設定 ----
OUTPUT_DIR  = Path(__file__).parent / "output"
SIM_HZ      = 240
SETTLE_TIME = 1.5   # 球が着地するまで待つ秒数
IMG_W, IMG_H = 640, 480

# ---- HSVカラー範囲 ----
# OpenCV: H=0-179, S=0-255, V=0-255
COLOR_RANGES = {
    "red": [
        (np.array([0,  120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]), np.array([179, 255, 255])),   # 赤は色相が0と180の両端
    ],
    "blue": [
        (np.array([100, 120, 70]), np.array([130, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
}

# 描画色 (BGR)
DRAW_COLOR = {
    "red":    (0,   0,   220),
    "blue":   (220, 80,   0),
    "yellow": (0,   200, 220),
}

# ---- 球体定義 ----
SPHERES = {
    "red":    {"rgba": [1.0, 0.0, 0.0, 1.0], "pos": [ 0.8,  0.3, 0.8]},
    "blue":   {"rgba": [0.0, 0.4, 1.0, 1.0], "pos": [ 0.0,  0.0, 0.8]},
    "yellow": {"rgba": [1.0, 0.9, 0.0, 1.0], "pos": [-0.8, -0.3, 0.8]},
}


# ============================
#  PyBullet セットアップ
# ============================

def setup_world() -> tuple[int, dict]:
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    sphere_ids = {}
    for name, cfg in SPHERES.items():
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3, physicsClientId=client)
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.3,
            rgbaColor=cfg["rgba"], physicsClientId=client
        )
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=cfg["pos"],
            physicsClientId=client,
        )
        sphere_ids[name] = body_id
    return client, sphere_ids


def settle(client: int, seconds: float):
    """重力で球が着地するまでシミュレーションを進める。"""
    steps = int(seconds * SIM_HZ)
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client)


# ============================
#  カメラキャプチャ
# ============================

def capture_bgr(client: int) -> np.ndarray:
    """俯瞰カメラで撮影してBGR画像(H×W×3)を返す。"""
    view_mat = p.computeViewMatrix(
        cameraEyePosition   =[0, 0, 4.0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector      =[0, 1, 0],
        physicsClientId=client,
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=IMG_W / IMG_H,
        nearVal=0.1, farVal=20,
        physicsClientId=client,
    )
    _, _, rgba_px, _, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    rgba = np.array(rgba_px, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
    bgr  = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return bgr


# ============================
#  色検出
# ============================

def detect_color(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """指定色のマスク画像を返す。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color_name]:
        mask |= cv2.inRange(hsv, lo, hi)
    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_objects(bgr: np.ndarray) -> list[dict]:
    """全色を検出してオブジェクトリストを返す。"""
    hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    results = []

    for color_name in COLOR_RANGES:
        mask       = detect_color(hsv, color_name)
        contours, _= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:   # 小さいノイズを除外
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            (ec_x, ec_y), ec_r = cv2.minEnclosingCircle(cnt)
            results.append({
                "color":  color_name,
                "area":   area,
                "bbox":   (x, y, w, h),
                "center": (cx, cy),
                "circle": (int(ec_x), int(ec_y), int(ec_r)),
            })

    return results


# ============================
#  描画
# ============================

def draw_results(bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for obj in objects:
        color = DRAW_COLOR[obj["color"]]
        # バウンディングボックス
        x, y, w, h = obj["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        # 外接円
        cx, cy, cr = obj["circle"]
        cv2.circle(out, (cx, cy), cr, color, 2)
        # ラベル
        label = f"{obj['color']} ({obj['area']:.0f}px)"
        cv2.putText(out, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # 中心点
        cv2.drawMarker(out, obj["center"], color,
                       cv2.MARKER_CROSS, 15, 2)
    return out


# ============================
#  保存
# ============================

def save_images(raw: np.ndarray, annotated: np.ndarray, masks: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = int(time.time())

    raw_path  = OUTPUT_DIR / f"raw_{ts}.png"
    ann_path  = OUTPUT_DIR / f"annotated_{ts}.png"
    cv2.imwrite(str(raw_path),  raw)
    cv2.imwrite(str(ann_path),  annotated)
    print(f"[SAVE] raw      → {raw_path}")
    print(f"[SAVE] annotated→ {ann_path}")

    for name, mask in masks.items():
        p_mask = OUTPUT_DIR / f"mask_{name}_{ts}.png"
        cv2.imwrite(str(p_mask), mask)
        print(f"[SAVE] mask_{name:6s}→ {p_mask}")


# ============================
#  メイン
# ============================

def main():
    print("=== Phase 2: OpenCV 色認識 ===\n")

    # 1. シミュレーション構築
    client, sphere_ids = setup_world()
    print(f"[SIM] 球体配置完了: {list(sphere_ids.keys())}")
    settle(client, SETTLE_TIME)
    print(f"[SIM] {SETTLE_TIME}s 着地待ち完了\n")

    # 2. PyBullet 実際座標を記録
    print("[SIM] 球体の実際座標:")
    for name, body_id in sphere_ids.items():
        pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
        print(f"  {name:6s} → 3D位置 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # 3. カメラキャプチャ
    raw_bgr = capture_bgr(client)
    print(f"\n[CAM] キャプチャ完了: {raw_bgr.shape[1]}x{raw_bgr.shape[0]} BGR")

    # 4. 色検出
    hsv   = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    masks = {c: detect_color(hsv, c) for c in COLOR_RANGES}
    objects = find_objects(raw_bgr)

    print(f"\n[DETECT] 検出結果: {len(objects)} 個のオブジェクト")
    print("-" * 50)
    for obj in objects:
        print(f"  色: {obj['color']:6s} | 面積: {obj['area']:6.0f}px | "
              f"中心: {obj['center']} | bbox: {obj['bbox']}")
    print("-" * 50)

    # 5. 描画 & 保存
    annotated = draw_results(raw_bgr, objects)
    save_images(raw_bgr, annotated, masks)

    # 6. 未検出の色を警告
    detected_colors = {o["color"] for o in objects}
    for c in COLOR_RANGES:
        if c not in detected_colors:
            print(f"[WARN] '{c}' が検出されませんでした（HSV範囲調整が必要かもしれません）")

    p.disconnect(physicsClientId=client)
    print("\n[INFO] Phase 2 完了。output/ フォルダを確認してください。")


if __name__ == "__main__":
    main()
