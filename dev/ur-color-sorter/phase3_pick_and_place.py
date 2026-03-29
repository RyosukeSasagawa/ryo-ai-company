"""
Phase 3: Rule-based Pick & Place
- Kuka iiwa7 アームが色認識結果をもとに球を仕分けボックスへ運ぶ
- 赤→左ボックス / 青→中央ボックス / 黄→右ボックス
- IK制御 + Constraintグリップ（物理ベース）
- output/ にフレーム画像 + animated GIF を保存
"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Phase 2 の色検出関数を再利用
from phase2_color_detection import detect_color, find_objects, COLOR_RANGES

# ============================================================
#  定数
# ============================================================
OUTPUT_DIR   = Path(__file__).parent / "output"
SIM_HZ       = 240
END_EFF_IDX  = 6      # Kuka iiwa7 のエンドエフェクタリンクインデックス
NUM_JOINTS   = 7
MOVE_STEPS   = 180    # 1動作フェーズあたりのステップ数
FRAME_SKIP   = 12     # N ステップごとに1フレーム保存
IMG_W        = 640
IMG_H        = 480
BALL_RADIUS  = 0.10
PREGRASP_Z   = 0.55   # アプローチ高さ
GRASP_Z_OFF  = 0.14   # エンドエフェクタと球中心のZ オフセット（グリップ時）

# 球の初期配置
BALL_CONFIG = {
    "red":    {"rgba": [1.0, 0.0, 0.0, 1.0], "pos": [0.45,  0.25, BALL_RADIUS]},
    "blue":   {"rgba": [0.0, 0.4, 1.0, 1.0], "pos": [0.55,  0.00, BALL_RADIUS]},
    "yellow": {"rgba": [1.0, 0.9, 0.0, 1.0], "pos": [0.45, -0.25, BALL_RADIUS]},
}

# 仕分け先ボックス（各色）
DESTINATIONS = {
    "red":    {"xy": [0.10,  0.62], "rgba": [0.9, 0.3, 0.3, 0.6], "label": "LEFT  (Red)"},
    "blue":   {"xy": [0.65,  0.00], "rgba": [0.3, 0.5, 1.0, 0.6], "label": "CENTER(Blue)"},
    "yellow": {"xy": [0.10, -0.62], "rgba": [1.0, 1.0, 0.3, 0.6], "label": "RIGHT (Yellow)"},
}

# アニメーション用カメラ（斜め俯瞰）
CAM_EYE    = [1.3, 1.3, 1.1]
CAM_TARGET = [0.3, 0.0, 0.15]
CAM_UP     = [0.0, 0.0, 1.0]


# ============================================================
#  ワールド構築
# ============================================================

def setup_world() -> tuple:
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    return client


def load_robot(client: int) -> int:
    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client,
    )
    # 初期姿勢をホームポジションに
    home = [0, 0.4, 0, -1.2, 0, 0.8, 0]
    for j, angle in enumerate(home):
        p.resetJointState(robot_id, j, angle, physicsClientId=client)
    print(f"[ROBOT] Kuka iiwa7 ロード完了 (id={robot_id})")
    return robot_id


def spawn_balls(client: int) -> dict:
    ids = {}
    for name, cfg in BALL_CONFIG.items():
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                       physicsClientId=client)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                                     rgbaColor=cfg["rgba"], physicsClientId=client)
        bid = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=cfg["pos"],
            physicsClientId=client,
        )
        ids[name] = bid
        print(f"[BALL]  {name:6s} → body_id={bid}, pos={cfg['pos']}")
    return ids


def spawn_boxes(client: int) -> dict:
    ids = {}
    half = [0.14, 0.14, 0.02]
    for name, cfg in DESTINATIONS.items():
        x, y = cfg["xy"]
        shape  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                        physicsClientId=client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                     rgbaColor=cfg["rgba"], physicsClientId=client)
        bid = p.createMultiBody(
            baseMass=0,                        # 静的
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, 0.02],
            physicsClientId=client,
        )
        ids[name] = bid
        print(f"[BOX]   {cfg['label']} → body_id={bid}, pos=[{x}, {y}, 0.02]")
    return ids


def settle(client: int, seconds: float = 0.5):
    for _ in range(int(seconds * SIM_HZ)):
        p.stepSimulation(physicsClientId=client)


# ============================================================
#  アーム制御
# ============================================================

def move_arm(client: int, robot_id: int, target_pos: list,
             steps: int = MOVE_STEPS, frames: list = None):
    """IK でアームをターゲット位置へ移動。frames に渡すとフレーム収集。"""
    joint_angles = p.calculateInverseKinematics(
        robot_id, END_EFF_IDX, target_pos,
        physicsClientId=client,
    )
    for step in range(steps):
        for j in range(NUM_JOINTS):
            p.setJointMotorControl2(
                robot_id, j, p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=500,
                physicsClientId=client,
            )
        p.stepSimulation(physicsClientId=client)
        if frames is not None and step % FRAME_SKIP == 0:
            frames.append(capture_frame(client))


def get_ee_pos(client: int, robot_id: int) -> np.ndarray:
    state = p.getLinkState(robot_id, END_EFF_IDX, physicsClientId=client)
    return np.array(state[4])   # worldLinkFramePosition


# ============================================================
#  グリップ / リリース
# ============================================================

def grip(client: int, robot_id: int, ball_id: int) -> int:
    cid = p.createConstraint(
        robot_id, END_EFF_IDX,
        ball_id, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, GRASP_Z_OFF],   # EEフレームから少し下
        [0, 0, 0],
        physicsClientId=client,
    )
    return cid


def release(client: int, constraint_id: int):
    p.removeConstraint(constraint_id, physicsClientId=client)


# ============================================================
#  フレーム / GIF
# ============================================================

def capture_frame(client: int) -> np.ndarray:
    view = p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(
        fov=55, aspect=IMG_W / IMG_H,
        nearVal=0.1, farVal=10,
        physicsClientId=client,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        IMG_W, IMG_H, view, proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[..., :3]
    return rgb   # RGB


def save_gif(frames: list, path: Path, duration_ms: int = 60):
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[GIF]  {path}  ({len(pil_frames)} フレーム)")


# ============================================================
#  色検出（Phase 2 統合）
# ============================================================

def detect_balls_in_scene(client: int) -> dict:
    """カメラ画像から色を検出してラベルを返す。"""
    # 真上カメラ（Phase 2 と同じ設定）
    view = p.computeViewMatrix(
        [0, 0, 3.5], [0, 0, 0], [0, 1, 0],
        physicsClientId=client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=IMG_W / IMG_H,
        nearVal=0.1, farVal=10,
        physicsClientId=client,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        IMG_W, IMG_H, view, proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    rgba_arr = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
    bgr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    return find_objects(bgr)


# ============================================================
#  ピック＆プレース シーケンス
# ============================================================

def pick_and_place(client: int, robot_id: int, ball_id: int,
                   ball_pos: list, dest_xy: list,
                   color: str, label: str, frames: list):
    bx, by, bz = ball_pos
    dx, dy     = dest_xy
    drop_z     = 0.25    # リリース高さ

    print(f"\n  ▶ [{color.upper()}] {label}")
    print(f"    球位置: ({bx:.2f}, {by:.2f}, {bz:.2f})  →  Box: ({dx:.2f}, {dy:.2f})")

    # 1. プリグラスプ（球の真上）
    print(f"    [1/5] プリグラスプ位置へ移動...")
    move_arm(client, robot_id, [bx, by, PREGRASP_Z], frames=frames)

    # 2. グラスプ位置へ降下
    print(f"    [2/5] 降下してグリップ...")
    move_arm(client, robot_id, [bx, by, bz + GRASP_Z_OFF], steps=120, frames=frames)
    settle(client, 0.1)
    cid = grip(client, robot_id, ball_id)
    print(f"          Constraint 生成 (id={cid})")

    # 3. 持ち上げ
    print(f"    [3/5] 持ち上げ...")
    move_arm(client, robot_id, [bx, by, PREGRASP_Z], frames=frames)

    # 4. 目的ボックス上へ移動
    print(f"    [4/5] ボックス上へ移動...")
    move_arm(client, robot_id, [dx, dy, PREGRASP_Z], frames=frames)

    # 5. リリース
    print(f"    [5/5] リリース...")
    move_arm(client, robot_id, [dx, dy, drop_z], steps=120, frames=frames)
    settle(client, 0.1)
    release(client, cid)
    settle(client, 0.3)

    # リリース後の球位置確認
    final_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
    print(f"          完了 → 球の最終位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")

    # アームを少し引く
    move_arm(client, robot_id, [dx, dy, PREGRASP_Z], steps=80, frames=frames)


# ============================================================
#  メイン
# ============================================================

def main():
    print("=" * 60)
    print("  Phase 3: Rule-based Pick & Place")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    frames: list = []

    # ---- ワールド構築 ----
    print("\n[SETUP] ワールド構築中...")
    client  = setup_world()
    robot_id = load_robot(client)
    ball_ids = spawn_balls(client)
    box_ids  = spawn_boxes(client)

    print("\n[SETUP] 初期化（着地待ち）...")
    settle(client, 1.0)
    frames.append(capture_frame(client))

    # ---- Phase 2 色検出 ----
    print("\n[DETECT] Phase 2 色認識で球を識別中...")
    detected = detect_balls_in_scene(client)
    print(f"         {len(detected)} 個のオブジェクトを検出:")
    for obj in detected:
        print(f"         - {obj['color']:6s} | center={obj['center']} | area={obj['area']:.0f}px")

    detected_colors = {o["color"] for o in detected}
    for c in COLOR_RANGES:
        if c not in detected_colors:
            print(f"[WARN]   '{c}' が画像から検出されませんでした")

    # ---- ルールベース仕分け ----
    print("\n[PLAN]  仕分けルール:")
    for color, dest in DESTINATIONS.items():
        print(f"         {color:6s} → {dest['label']}")

    ORDER = ["red", "blue", "yellow"]   # 処理順

    print("\n[START] ピック＆プレース開始")
    print("-" * 60)

    for color in ORDER:
        if color not in ball_ids:
            continue
        ball_id = ball_ids[color]
        dest    = DESTINATIONS[color]

        # 球の現在3D座標を取得（PyBulletから直接）
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
        pick_and_place(
            client, robot_id,
            ball_id, list(pos), dest["xy"],
            color, dest["label"], frames,
        )

    # ---- 最終状態 ----
    settle(client, 0.5)
    for _ in range(3):
        frames.append(capture_frame(client))

    print("\n" + "-" * 60)
    print("[DONE]  全球の仕分け完了!\n")
    print("[RESULT] 最終球位置:")
    for color, bid in ball_ids.items():
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=client)
        dx, dy = DESTINATIONS[color]["xy"]
        dist   = ((pos[0]-dx)**2 + (pos[1]-dy)**2)**0.5
        print(f"  {color:6s} → ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
              f"  ボックスまでの距離: {dist:.3f}m")

    # ---- 画像保存 ----
    import time
    ts = int(time.time())

    # 最終スナップショット
    snap_path = OUTPUT_DIR / f"phase3_final_{ts}.png"
    final_bgr = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(snap_path), final_bgr)
    print(f"\n[SAVE]  最終スナップショット → {snap_path}")

    # 個別フレーム
    frame_dir = OUTPUT_DIR / f"phase3_frames_{ts}"
    frame_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    print(f"[SAVE]  フレーム {len(frames)} 枚 → {frame_dir}/")

    # アニメーション GIF
    gif_path = OUTPUT_DIR / f"phase3_animation_{ts}.gif"
    save_gif(frames, gif_path, duration_ms=60)

    p.disconnect(physicsClientId=client)
    print("\n[INFO]  Phase 3 完了。")


if __name__ == "__main__":
    main()
