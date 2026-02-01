import gymnasium as gym
import gym_ligo
import os
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# REMOVED: VecNormalize is no longer needed!

def main():
    # --- 1. CONFIG ---
    TOTAL_TIMESTEPS = 500_000
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 2. CREATE ENVIRONMENT ---
    def make_env():
        # Clean gym.make usage
        env = gym.make("Ligo-v0", render_mode=None)
        env = Monitor(env, LOG_DIR)
        return env

    env = DummyVecEnv([make_env])

    # NOTE: VecNormalize is gone. The environment now maps:
    # Action 1.0 -> 10 uN
    # Obs 1.0 -> 1 Picometer

    # --- 3. DEFINE THE AGENT ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=LOG_DIR,
        device="auto"
    )

    # --- 4. CALLBACKS ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_ligo"
    )

    print("Starting training (Internal Scaling Mode)...")

    # --- 5. TRAIN ---
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted... saving model.")

    # --- 6. SAVE MODEL ---
    model.save(f"{MODEL_DIR}/ppo_ligo_final")
    # REMOVED: env.save(...) for normalization stats is no longer needed

    print("Training finished.")

if __name__ == "__main__":
    main()