"""
Phase 1 テスト: PyBullet 動作確認
- 地面 + 色違いの球体（赤・青・黄）を配置
- 重力をかけて落下シミュレーション
- GUI使用不可（WSL2等）の場合はDIRECTモードでターミナル出力
"""

import pybullet as p
import pybullet_data
import time


# ---- 色定義 (RGBA) ----
COLORS = {
    "red":    ([1.0, 0.0, 0.0, 1.0], [1.5, 0.0, 1.0]),   # 色, 初期位置
    "blue":   ([0.0, 0.4, 1.0, 1.0], [0.0, 0.0, 2.0]),
    "yellow": ([1.0, 0.9, 0.0, 1.0], [-1.5, 0.0, 3.0]),
}

SIM_DURATION = 5.0   # シミュレーション時間（秒）
STEP_HZ      = 240   # シミュレーション周波数


def connect(use_gui: bool) -> int:
    """PyBulletに接続。失敗したらDIRECTモードにフォールバック。"""
    if use_gui:
        try:
            client = p.connect(p.GUI)
            print("[INFO] GUIモードで起動しました。")
            return client
        except Exception as e:
            print(f"[WARN] GUI起動失敗 ({e})。DIRECTモードに切り替えます。")
    client = p.connect(p.DIRECT)
    print("[INFO] DIRECTモード（ヘッドレス）で起動しました。")
    return client


def setup_world(client: int) -> dict:
    """地面と球体を配置してIDを返す。"""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)

    # 地面
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)

    # 球体を生成
    sphere_ids = {}
    for name, (rgba, pos) in COLORS.items():
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3, physicsClientId=client)
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.3, rgbaColor=rgba, physicsClientId=client
        )
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=pos,
            physicsClientId=client,
        )
        sphere_ids[name] = body_id
        print(f"[SETUP] {name:6s} 球 → body_id={body_id}, 初期位置={pos}")

    return {"plane": plane_id, "spheres": sphere_ids}


def run_simulation(client: int, sphere_ids: dict):
    """シミュレーションを SIM_DURATION 秒間実行してターミナルに座標を出力。"""
    total_steps = int(SIM_DURATION * STEP_HZ)
    report_interval = STEP_HZ // 4  # 0.25秒ごとに出力

    print(f"\n[SIM] 開始 — {SIM_DURATION}秒間 / {total_steps}ステップ")
    print("-" * 55)

    start = time.time()

    for step in range(total_steps):
        p.stepSimulation(physicsClientId=client)

        if step % report_interval == 0:
            t = step / STEP_HZ
            parts = [f"t={t:.2f}s"]
            for name, body_id in sphere_ids.items():
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
                parts.append(f"{name}=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
            print("  " + "  |  ".join(parts))

        # GUIモード時はリアルタイムに近いペースで進める
        if p.getConnectionInfo(client)["connectionMethod"] == p.GUI:
            time.sleep(1.0 / STEP_HZ)

    elapsed = time.time() - start
    print("-" * 55)
    print(f"[SIM] 完了 — 実経過時間: {elapsed:.2f}s")


def main():
    client = connect(use_gui=True)

    try:
        world = setup_world(client)
        run_simulation(client, world["spheres"])

        # 最終位置サマリー
        print("\n[RESULT] 最終位置サマリー:")
        for name, body_id in world["spheres"].items():
            pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
            vel, _ = p.getBaseVelocity(body_id, physicsClientId=client)
            print(f"  {name:6s}  位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                  f"速度=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")

    finally:
        p.disconnect(physicsClientId=client)
        print("\n[INFO] PyBullet 切断。Phase 1 テスト完了。")


if __name__ == "__main__":
    main()
