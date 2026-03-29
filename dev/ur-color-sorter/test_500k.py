"""
500k モデル テスト専用スクリプト
- models/ppo_pickplace_500k.zip をロードしてテストのみ実行
- 学習は一切しない
- deterministic=False（確率的行動）でテスト
- GIF出力: output/phase4_demo_500k_test.gif
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image

from stable_baselines3 import PPO

from phase4_reinforcement_learning import PickPlaceEnv

# ============================================================
#  設定
# ============================================================
N_TEST_EPISODES = 5

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_PATH = MODELS_DIR / "ppo_pickplace_500k.zip"
GIF_PATH   = OUTPUT_DIR / "phase4_demo_500k_test.gif"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
#  テスト & GIF
# ============================================================

def test_and_gif():
    if not MODEL_PATH.exists():
        print(f"[ERROR] モデルが見つかりません: {MODEL_PATH}", flush=True)
        return

    print(f"[LOAD]  モデルをロード: {MODEL_PATH}", flush=True)
    model = PPO.load(str(MODEL_PATH), device="cpu")
    print(f"[LOAD]  完了", flush=True)

    print(f"\n[TEST]  {N_TEST_EPISODES} エピソードテスト（確率的: deterministic=False）", flush=True)
    print("─" * 60, flush=True)

    env = PickPlaceEnv(render_mode="rgb_array")
    all_frames: list[np.ndarray] = []
    results = []

    for ep in range(1, N_TEST_EPISODES + 1):
        obs, _ = env.reset()
        done = truncated = False
        total_r = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            if env._step_cnt % 4 == 0:
                all_frames.append(env.render())

        success = done and not truncated
        results.append(success)
        print(f"  Ep {ep}: reward={total_r:+.3f} | steps={env._step_cnt:3d} | "
              f"{'✅ 成功' if success else '❌ 失敗'}", flush=True)

    env.close()

    # GIF 保存
    if all_frames:
        pil = [Image.fromarray(f) for f in all_frames]
        pil[0].save(GIF_PATH, save_all=True, append_images=pil[1:], duration=50, loop=0)
        print(f"\n[GIF]   {GIF_PATH}  ({len(pil)} フレーム)", flush=True)

    suc_rate = sum(results) / len(results) * 100
    print(f"[RESULT] 成功率: {suc_rate:.0f}% ({sum(results)}/{len(results)})", flush=True)


# ============================================================
#  メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print(f"  UR Color Sorter — PPO 500k テスト（学習なし）", flush=True)
    print(f"  開始: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 72, flush=True)

    test_and_gif()

    print(f"\n[DONE]  完了: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"        GIF:   {GIF_PATH}", flush=True)
