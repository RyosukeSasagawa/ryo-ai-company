"""
Phase 4: 強化学習 Pick & Place
- Gymnasium カスタム環境 (PickPlaceEnv)
- 仮想 sticky gripper（IK不要で高速）
- PPO (Stable-Baselines3) 100,000ステップ学習
- 学習済みモデル → models/ppo_pickplace.zip
- テスト実行 → output/phase4_demo.gif
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image

# ============================================================
#  定数
# ============================================================
MODELS_DIR      = Path(__file__).parent / "models"
OUTPUT_DIR      = Path(__file__).parent / "output"

TOTAL_TIMESTEPS = 100_000
LOG_INTERVAL    = 5_000
MAX_EP_STEPS    = 300
PHYS_STEPS      = 10       # 1アクションあたりの物理ステップ数
ACT_SCALE       = 0.04     # 1アクションの最大移動量(m)

# ワークスペース境界
WS_LOW  = np.array([0.10, -0.60, 0.04], dtype=np.float32)
WS_HIGH = np.array([0.90,  0.60, 0.70], dtype=np.float32)

# 目標ボックス（固定）
TARGET_XY     = np.array([0.65, 0.40], dtype=np.float32)
TARGET_RADIUS = 0.14    # 成功判定半径(m)
SUCCESS_Z     = 0.25    # 成功判定Z上限(m)

# 球の生成範囲（ランダム）
BALL_X = (0.25, 0.65)
BALL_Y = (-0.35, 0.35)
BALL_R = 0.07


# ============================================================
#  Gymnasium カスタム環境
# ============================================================

class PickPlaceEnv(gym.Env):
    """
    観測 (11次元):
      EE位置(3) | 球位置(3) | 目標XY(2) | 把持フラグ(1) | EE-球距離(1) | 球-目標距離(1)
    行動 (4次元, 連続 [-1, 1]):
      [Δx, Δy, Δz, grip]  grip>0 → 把持試行
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._client    = None
        self._ball_id   = None
        self._grip_id   = None
        self._ee_pos    = np.zeros(3, dtype=np.float32)
        self._ball_pos  = np.zeros(3, dtype=np.float32)
        self._has_ball  = False
        self._step_cnt  = 0

        self._init_world()

    # ---- ワールド構築 ----

    def _init_world(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # 仮想グリッパー（mass=0 で kinematic）
        g_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05,
                                        physicsClientId=self._client)
        g_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                     rgbaColor=[0.7, 0.7, 0.7, 0.9],
                                     physicsClientId=self._client)
        self._grip_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=g_col,
            baseVisualShapeIndex=g_vis,
            basePosition=[0.5, 0.0, 0.4],
            physicsClientId=self._client,
        )

        # 目標ボックス（静的）
        tx, ty = TARGET_XY
        b_col = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[TARGET_RADIUS, TARGET_RADIUS, 0.02],
                                        physicsClientId=self._client)
        b_vis = p.createVisualShape(p.GEOM_BOX,
                                     halfExtents=[TARGET_RADIUS, TARGET_RADIUS, 0.02],
                                     rgbaColor=[0.2, 0.85, 0.3, 0.65],
                                     physicsClientId=self._client)
        p.createMultiBody(0, b_col, b_vis, [tx, ty, 0.02], physicsClientId=self._client)

    def _spawn_ball(self):
        """古い球を削除して新しい球をランダム位置に生成。"""
        if self._ball_id is not None:
            p.removeBody(self._ball_id, physicsClientId=self._client)
            self._ball_id = None

        bx = float(self.np_random.uniform(*BALL_X))
        by = float(self.np_random.uniform(*BALL_Y))
        s = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R, physicsClientId=self._client)
        v = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R,
                                 rgbaColor=[0.95, 0.2, 0.2, 1.0],
                                 physicsClientId=self._client)
        self._ball_id = p.createMultiBody(
            baseMass=0.15, baseCollisionShapeIndex=s,
            baseVisualShapeIndex=v,
            basePosition=[bx, by, BALL_R],
            physicsClientId=self._client,
        )
        for _ in range(30):
            p.stepSimulation(physicsClientId=self._client)

    # ---- Gymnasium API ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._has_ball = False
        self._step_cnt = 0
        self._ee_pos   = np.array([0.5, 0.0, 0.4], dtype=np.float32)

        p.resetBasePositionAndOrientation(
            self._grip_id, self._ee_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self._client,
        )
        self._spawn_ball()
        self._sync_ball_pos()
        return self._obs(), {}

    def step(self, action):
        self._step_cnt += 1
        act = np.clip(action.astype(np.float32), -1.0, 1.0)
        dx, dy, dz, grip_act = act

        # ---- EE 移動 ----
        self._ee_pos = np.clip(
            self._ee_pos + np.array([dx, dy, dz], dtype=np.float32) * ACT_SCALE,
            WS_LOW, WS_HIGH,
        )
        p.resetBasePositionAndOrientation(
            self._grip_id, self._ee_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self._client,
        )

        # ---- グリップ制御（sticky gripper）----
        dist_to_ball = float(np.linalg.norm(self._ee_pos - self._ball_pos))

        if grip_act > 0 and not self._has_ball and dist_to_ball < (BALL_R + 0.06):
            # 把持: 球をEEに固定（テレポート方式）
            self._has_ball = True

        elif grip_act <= 0 and self._has_ball:
            # 解放
            self._has_ball = False
            # 解放時に下向き速度を与えてリアルに落下させる
            p.resetBaseVelocity(
                self._ball_id, [0, 0, -0.5], [0, 0, 0],
                physicsClientId=self._client,
            )

        if self._has_ball:
            # 球をEEと同位置に追従（テレポート）
            p.resetBasePositionAndOrientation(
                self._ball_id, self._ee_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self._client,
            )
            p.resetBaseVelocity(
                self._ball_id, [0, 0, 0], [0, 0, 0],
                physicsClientId=self._client,
            )

        # ---- 物理ステップ ----
        for _ in range(PHYS_STEPS):
            p.stepSimulation(physicsClientId=self._client)

        self._sync_ball_pos()

        # ---- 報酬・終了判定 ----
        reward, terminated = self._compute_reward(dist_to_ball)
        truncated = self._step_cnt >= MAX_EP_STEPS

        return self._obs(), reward, terminated, truncated, {}

    def _compute_reward(self, dist_to_ball: float) -> tuple[float, bool]:
        r = -0.001  # 時間ペナルティ

        dist_target = float(np.linalg.norm(self._ball_pos[:2] - TARGET_XY))

        if not self._has_ball:
            # 球へ接近する報酬（密な報酬）
            r += max(0.0, 0.30 - dist_to_ball) * 0.15

            # 成功判定: 球がボックス内に収まっている
            if dist_target < TARGET_RADIUS and self._ball_pos[2] < SUCCESS_Z:
                return r + 1.0, True

        else:
            # 把持中: 目標への接近報酬
            r += max(0.0, 0.60 - dist_target) * 0.10

        # 失敗: 球が落下
        if self._ball_pos[2] < -0.05:
            return r - 0.1, True

        return r, False

    def _obs(self) -> np.ndarray:
        dist_ball   = float(np.linalg.norm(self._ee_pos - self._ball_pos))
        dist_target = float(np.linalg.norm(self._ball_pos[:2] - TARGET_XY))
        return np.concatenate([
            self._ee_pos,
            self._ball_pos,
            TARGET_XY,
            [float(self._has_ball), dist_ball, dist_target],
        ]).astype(np.float32)

    def _sync_ball_pos(self):
        self._ball_pos = np.array(
            p.getBasePositionAndOrientation(self._ball_id, physicsClientId=self._client)[0],
            dtype=np.float32,
        )

    def render(self) -> np.ndarray:
        view = p.computeViewMatrix(
            [1.0, 0.9, 0.85], [0.45, 0.1, 0.15], [0, 0, 1],
            physicsClientId=self._client,
        )
        proj = p.computeProjectionMatrixFOV(
            55, 640 / 480, 0.1, 10,
            physicsClientId=self._client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            640, 480, view, proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        return np.array(rgba, dtype=np.uint8).reshape(480, 640, 4)[..., :3]

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


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
            # 成功エピソード判定（報酬 > 0.8 を目安）
            if self._cur_reward > 0.8:
                self._success += 1
            self._cur_reward = 0.0

        if self.n_calls % self._interval == 0 and self.n_calls > 0:
            recent   = self._ep_rewards[-20:] if self._ep_rewards else [0.0]
            mean_r   = float(np.mean(recent))
            suc_rate = self._success / max(1, self._ep_count) * 100
            elapsed  = time.time() - self._t0
            print(
                f"  Step {self.n_calls:7,}/{TOTAL_TIMESTEPS:,}"
                f" | Ep: {self._ep_count:5d}"
                f" | Reward(last20): {mean_r:+.4f}"
                f" | 成功率: {suc_rate:5.1f}%"
                f" | {elapsed:.0f}s"
            )
        return True


# ============================================================
#  学習
# ============================================================

def train() -> PPO:
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[TRAIN] 環境を初期化中...")
    env = DummyVecEnv([lambda: Monitor(PickPlaceEnv())])

    model = PPO(
        "MlpPolicy", env,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        learning_rate = 3e-4,
        ent_coef      = 0.02,    # 探索を促進
        clip_range    = 0.2,
        verbose       = 0,
        device        = "cpu",   # WSL2環境でCUDA不整合を回避
    )

    print(f"[TRAIN] PPO 学習開始 — {TOTAL_TIMESTEPS:,} ステップ")
    print(f"        policy=MLP | n_steps={model.n_steps} | "
          f"batch={model.batch_size} | epochs={model.n_epochs} | "
          f"lr={model.learning_rate} | ent={model.ent_coef}")
    print(f"{'─' * 72}")

    t0 = time.time()
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = TrainLogger(LOG_INTERVAL),
    )
    elapsed = time.time() - t0

    print(f"{'─' * 72}")
    print(f"[TRAIN] 完了 — 所要時間: {elapsed:.1f}s "
          f"({elapsed/60:.1f}min) / {TOTAL_TIMESTEPS/elapsed:.0f} steps/s")

    save_path = MODELS_DIR / "ppo_pickplace.zip"
    model.save(str(save_path))
    print(f"[SAVE]  モデル保存 → {save_path}")

    env.close()
    return model


# ============================================================
#  テスト & GIF 出力
# ============================================================

def test_and_gif(model: PPO, n_episodes: int = 3):
    print(f"\n[TEST]  学習済みモデルで {n_episodes} エピソードテスト（決定論的）")
    print(f"{'─' * 60}")

    env = PickPlaceEnv(render_mode="rgb_array")
    all_frames: list[np.ndarray] = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = truncated = False
        total_r = 0.0
        ep_frames: list[np.ndarray] = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            if env._step_cnt % 4 == 0:
                ep_frames.append(env.render())

        success = done and not truncated
        all_frames.extend(ep_frames)
        print(f"  Episode {ep}: reward={total_r:+.3f} | "
              f"steps={env._step_cnt:3d} | "
              f"{'✅ 成功' if success else '❌ 失敗'}")

    env.close()

    if not all_frames:
        print("[WARN]  フレームが空です。")
        return

    gif_path = OUTPUT_DIR / "phase4_demo.gif"
    pil_frames = [Image.fromarray(f) for f in all_frames]
    pil_frames[0].save(
        gif_path, save_all=True,
        append_images=pil_frames[1:], duration=50, loop=0,
    )
    print(f"\n[GIF]   {gif_path}  ({len(pil_frames)} フレーム)")


# ============================================================
#  メイン
# ============================================================

def main():
    print("=" * 72)
    print("  Phase 4: 強化学習 Pick & Place（PPO）")
    print("=" * 72)
    print()
    print("【設計サマリー】")
    print("  観測(11次元) : EE位置(3) + 球位置(3) + 目標XY(2) + 把持(1) + 距離×2")
    print("  行動(4次元)  : Δx, Δy, Δz（各±4cm）+ grip（>0で把持試行）")
    print("  グリッパー   : sticky gripper（IK不要・高速）")
    print("  報酬         : 接近報酬(密) / 把持後目標接近 / 成功+1.0 / ペナ-0.001/step")
    print(f"  学習ステップ : {TOTAL_TIMESTEPS:,} / エピソード最大: {MAX_EP_STEPS}step")
    print()

    model = train()
    test_and_gif(model, n_episodes=3)

    print("\n[INFO]  Phase 4 完了。")
    print(f"        モデル : {MODELS_DIR / 'ppo_pickplace.zip'}")
    print(f"        GIF   : {OUTPUT_DIR / 'phase4_demo.gif'}")


if __name__ == "__main__":
    main()
