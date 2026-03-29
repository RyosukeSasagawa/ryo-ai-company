"""
500k ステップ学習ランナー
- TOTAL_TIMESTEPS: 500,000
- モデル保存: models/ppo_pickplace_500k.zip
- GIF出力:   output/phase4_demo_500k.gif
- ログ:      標準出力（nohup で logs/training_500k.log にリダイレクト）
"""

import time
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# phase4 の環境クラスを再利用
from phase4_reinforcement_learning import PickPlaceEnv

# ============================================================
#  設定
# ============================================================
TOTAL_TIMESTEPS = 500_000
LOG_INTERVAL    = 10_000
N_TEST_EPISODES = 5

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_PATH = MODELS_DIR / "ppo_pickplace_500k.zip"
GIF_PATH   = OUTPUT_DIR / "phase4_demo_500k.gif"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
#  学習ログ コールバック
# ============================================================

class TrainLogger(BaseCallback):
    def __init__(self, log_interval: int = LOG_INTERVAL):
        super().__init__()
        self._interval   = log_interval
        self._ep_rewards: list[float] = []
        self._cur_reward = 0.0
        self._ep_count   = 0
        self._success    = 0
        self._t0         = time.time()

    def _on_step(self) -> bool:
        self._cur_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_count += 1
            self._ep_rewards.append(self._cur_reward)
            if self._cur_reward > 0.8:
                self._success += 1
            self._cur_reward = 0.0

        if self.n_calls % self._interval == 0 and self.n_calls > 0:
            recent   = self._ep_rewards[-20:] if self._ep_rewards else [0.0]
            mean_r   = float(np.mean(recent))
            suc_rate = self._success / max(1, self._ep_count) * 100
            elapsed  = time.time() - self._t0
            msg = (
                f"[STEP {self.n_calls:7,}/{TOTAL_TIMESTEPS:,}]"
                f"  Ep: {self._ep_count:5d}"
                f"  Reward(last20): {mean_r:+.4f}"
                f"  成功率: {suc_rate:5.1f}%"
                f"  elapsed: {elapsed:.0f}s"
            )
            print(msg, flush=True)
        return True


# ============================================================
#  学習
# ============================================================

def train() -> PPO:
    print(f"[TRAIN] 環境初期化...", flush=True)
    env = DummyVecEnv([lambda: Monitor(PickPlaceEnv())])

    model = PPO(
        "MlpPolicy", env,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        learning_rate = 3e-4,
        ent_coef      = 0.02,
        clip_range    = 0.2,
        verbose       = 0,
        device        = "cpu",
    )

    print(f"[TRAIN] PPO 500k 学習開始", flush=True)
    print(f"        policy=MLP | n_steps={model.n_steps} | batch={model.batch_size}"
          f" | epochs={model.n_epochs} | lr={model.learning_rate}", flush=True)
    print("─" * 72, flush=True)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TrainLogger(LOG_INTERVAL))
    elapsed = time.time() - t0

    print("─" * 72, flush=True)
    print(f"[TRAIN] 完了 — {elapsed:.1f}s ({elapsed/60:.1f}min) / "
          f"{TOTAL_TIMESTEPS/elapsed:.0f} steps/s", flush=True)

    model.save(str(MODEL_PATH))
    print(f"[SAVE]  モデル → {MODEL_PATH}", flush=True)

    env.close()
    return model


# ============================================================
#  テスト & GIF
# ============================================================

def test_and_gif(model: PPO):
    print(f"\n[TEST]  {N_TEST_EPISODES} エピソードテスト（決定論的）", flush=True)
    print("─" * 60, flush=True)

    env = PickPlaceEnv(render_mode="rgb_array")
    all_frames: list[np.ndarray] = []
    results = []

    for ep in range(1, N_TEST_EPISODES + 1):
        obs, _ = env.reset()
        done = truncated = False
        total_r = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
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
    print(f"  UR Color Sorter — PPO 500k 学習", flush=True)
    print(f"  開始: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 72, flush=True)

    model = train()
    test_and_gif(model)

    print(f"\n[DONE]  完了: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"        モデル: {MODEL_PATH}", flush=True)
    print(f"        GIF:   {GIF_PATH}", flush=True)
