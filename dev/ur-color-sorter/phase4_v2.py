"""
Phase 4 v2: 強化学習 Pick & Place（改善版）

改善点（v1からの変更）:
  1. 接近報酬を1/10に削減（局所最適回避）
  2. 成功報酬 +1.0 → +10.0
  3. 把持成功ボーナス +0.5 追加
  4. 目標上空での解放ボーナス +0.5 追加
  5. 成功判定 SUCCESS_Z: 0.25 → 0.30 に緩和
  6. VecNormalize 追加（観測の正規化）
  7. ent_coef: 0.02 → 0.05（探索促進）
  8. 100k 動作確認モード
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PIL import Image

# ============================================================
#  定数
# ============================================================
MODELS_DIR      = Path(__file__).parent / "models"
OUTPUT_DIR      = Path(__file__).parent / "output"

TOTAL_TIMESTEPS = 100_000
LOG_INTERVAL    = 10_000
MAX_EP_STEPS    = 300
PHYS_STEPS      = 10
ACT_SCALE       = 0.04

# ワークスペース境界
WS_LOW  = np.array([0.10, -0.60, 0.04], dtype=np.float32)
WS_HIGH = np.array([0.90,  0.60, 0.70], dtype=np.float32)

# 目標ボックス（固定）
TARGET_XY     = np.array([0.65, 0.40], dtype=np.float32)
TARGET_RADIUS = 0.14
SUCCESS_Z     = 0.30    # v1: 0.25 → 0.30 に緩和

# 球の生成範囲
BALL_X = (0.25, 0.65)
BALL_Y = (-0.35, 0.35)
BALL_R = 0.07

# 把持判定距離
GRIP_DIST = BALL_R + 0.06

# 目標上空ボーナス判定（XY距離がこれ以下で解放したらボーナス）
RELEASE_BONUS_RADIUS = TARGET_RADIUS * 1.5   # 0.21m


# ============================================================
#  Gymnasium カスタム環境 v2
# ============================================================

class PickPlaceEnvV2(gym.Env):
    """
    観測 (11次元):
      EE位置(3) | 球位置(3) | 目標XY(2) | 把持フラグ(1) | EE-球距離(1) | 球-目標距離(1)

    行動 (4次元, 連続 [-1, 1]):
      [Δx, Δy, Δz, grip]  grip>0 → 把持試行

    報酬設計 v2:
      - 時間ペナルティ:         -0.001/step
      - 接近報酬(把持前):       max(0, 0.30 - dist) * 0.015  ← 1/10に削減
      - 把持成功ボーナス:        +0.5（把持した瞬間1回だけ）
      - 目標接近報酬(把持中):   max(0, 0.60 - dist_target) * 0.10
      - 目標上空での解放ボーナス: +0.5
      - 成功報酬:               +10.0
      - 落下ペナルティ:         -0.1
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

        self._client   = None
        self._ball_id  = None
        self._grip_id  = None
        self._ee_pos   = np.zeros(3, dtype=np.float32)
        self._ball_pos = np.zeros(3, dtype=np.float32)
        self._has_ball = False
        self._step_cnt = 0
        self._grip_bonus_given = False  # 把持ボーナスを1エピソードで1回だけ与える

        self._init_world()

    # ---- ワールド構築 ----

    def _init_world(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # 仮想グリッパー
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
        self._grip_bonus_given = False
        self._ee_pos = np.array([0.5, 0.0, 0.4], dtype=np.float32)

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

        dist_to_ball = float(np.linalg.norm(self._ee_pos - self._ball_pos))
        release_bonus = 0.0

        # ---- グリップ制御 ----
        if grip_act > 0 and not self._has_ball and dist_to_ball < GRIP_DIST:
            self._has_ball = True

        elif grip_act <= 0 and self._has_ball:
            # 解放
            dist_target_xy = float(np.linalg.norm(self._ball_pos[:2] - TARGET_XY))
            if dist_target_xy < RELEASE_BONUS_RADIUS:
                release_bonus = 0.5  # 目標上空での解放ボーナス
            self._has_ball = False
            p.resetBaseVelocity(
                self._ball_id, [0, 0, -0.5], [0, 0, 0],
                physicsClientId=self._client,
            )

        if self._has_ball:
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
        reward, terminated = self._compute_reward(dist_to_ball, release_bonus)
        truncated = self._step_cnt >= MAX_EP_STEPS

        return self._obs(), reward, terminated, truncated, {}

    def _compute_reward(self, dist_to_ball: float, release_bonus: float) -> tuple[float, bool]:
        r = -0.001  # 時間ペナルティ

        dist_target = float(np.linalg.norm(self._ball_pos[:2] - TARGET_XY))

        if not self._has_ball:
            # 接近報酬（1/10に削減）
            r += max(0.0, 0.30 - dist_to_ball) * 0.015

            # 目標上空での解放ボーナス
            r += release_bonus

            # 成功判定（SUCCESS_Z を 0.30 に緩和）
            if dist_target < TARGET_RADIUS and self._ball_pos[2] < SUCCESS_Z:
                return r + 10.0, True

        else:
            # 把持成功ボーナス（1エピソード1回のみ）
            if not self._grip_bonus_given:
                r += 0.5
                self._grip_bonus_given = True

            # 把持中: 目標への接近報酬
            r += max(0.0, 0.60 - dist_target) * 0.10

        # 落下ペナルティ
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
    def __init__(self, log_interval: int = LOG_INTERVAL, total: int = TOTAL_TIMESTEPS):
        super().__init__()
        self._interval   = log_interval
        self._total      = total
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
            # 成功エピソード判定（成功報酬+10を含むので9.0超えを目安）
            if self._cur_reward > 9.0:
                self._success += 1
            self._cur_reward = 0.0

        if self.n_calls % self._interval == 0 and self.n_calls > 0:
            recent   = self._ep_rewards[-20:] if self._ep_rewards else [0.0]
            mean_r   = float(np.mean(recent))
            suc_rate = self._success / max(1, self._ep_count) * 100
            elapsed  = time.time() - self._t0
            print(
                f"[STEP {self.n_calls:7,}/{self._total:,}]"
                f"  Ep: {self._ep_count:5d}"
                f"  Reward(last20): {mean_r:+.4f}"
                f"  成功率: {suc_rate:5.1f}%"
                f"  elapsed: {elapsed:.0f}s",
                flush=True,
            )
        return True


# ============================================================
#  学習
# ============================================================

def train() -> tuple[PPO, VecNormalize]:
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("[TRAIN] 環境初期化...", flush=True)
    env = DummyVecEnv([lambda: Monitor(PickPlaceEnvV2())])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        learning_rate = 3e-4,
        ent_coef      = 0.05,    # v1: 0.02 → 0.05（探索促進）
        clip_range    = 0.2,
        verbose       = 0,
        device        = "cpu",
    )

    print(f"[TRAIN] PPO v2 100k 学習開始", flush=True)
    print(
        f"        policy=MLP | n_steps={model.n_steps} | batch={model.batch_size}"
        f" | epochs={model.n_epochs} | lr={model.learning_rate}"
        f" | ent_coef={model.ent_coef}",
        flush=True,
    )
    print("─" * 72, flush=True)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TrainLogger(LOG_INTERVAL, TOTAL_TIMESTEPS))
    elapsed = time.time() - t0

    print("─" * 72, flush=True)
    print(
        f"[TRAIN] 完了 — {elapsed:.1f}s ({elapsed/60:.1f}min)"
        f" / {TOTAL_TIMESTEPS/elapsed:.0f} steps/s",
        flush=True,
    )

    model_path = MODELS_DIR / "ppo_pickplace_v2_100k.zip"
    vecnorm_path = MODELS_DIR / "ppo_pickplace_v2_100k_vecnorm.pkl"
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    print(f"[SAVE]  モデル    → {model_path}", flush=True)
    print(f"[SAVE]  VecNorm   → {vecnorm_path}", flush=True)

    env.close()
    return model, env


# ============================================================
#  テスト & GIF
# ============================================================

def test_and_gif(model: PPO, n_episodes: int = 5):
    print(f"\n[TEST]  {n_episodes} エピソードテスト（決定論的）", flush=True)
    print("─" * 60, flush=True)

    raw_env = PickPlaceEnvV2(render_mode="rgb_array")
    test_vec = DummyVecEnv([lambda: raw_env])

    vecnorm_path = MODELS_DIR / "ppo_pickplace_v2_100k_vecnorm.pkl"
    test_vec = VecNormalize.load(str(vecnorm_path), test_vec)
    test_vec.training = False
    test_vec.norm_reward = False

    all_frames: list[np.ndarray] = []
    results = []

    for ep in range(1, n_episodes + 1):
        obs = test_vec.reset()
        done = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = test_vec.step(action)
            total_r += float(reward[0])
            done = bool(dones[0])
            if raw_env._step_cnt % 4 == 0:
                all_frames.append(raw_env.render())

        success = raw_env._step_cnt < MAX_EP_STEPS
        results.append(success)
        print(
            f"  Ep {ep}: reward={total_r:+.3f} | steps={raw_env._step_cnt:3d} | "
            f"{'✅ 成功' if success else '❌ 失敗'}",
            flush=True,
        )

    test_vec.close()

    if all_frames:
        gif_path = OUTPUT_DIR / "phase4_v2_100k_demo.gif"
        pil = [Image.fromarray(f) for f in all_frames]
        pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=50, loop=0)
        print(f"\n[GIF]   {gif_path}  ({len(pil)} フレーム)", flush=True)

    suc_rate = sum(results) / len(results) * 100
    print(f"[RESULT] 成功率: {suc_rate:.0f}% ({sum(results)}/{len(results)})", flush=True)


# ============================================================
#  メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("  UR Color Sorter — PPO v2 100k 動作確認", flush=True)
    print("  改善: 報酬設計刷新 / VecNormalize / ent_coef=0.05", flush=True)
    print(f"  開始: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 72, flush=True)

    model, _ = train()
    test_and_gif(model)

    model_path   = MODELS_DIR / "ppo_pickplace_v2_100k.zip"
    gif_path     = OUTPUT_DIR / "phase4_v2_100k_demo.gif"
    print(f"\n[DONE]  完了: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"        モデル: {model_path}", flush=True)
    print(f"        GIF:   {gif_path}", flush=True)
