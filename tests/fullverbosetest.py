import numpy as np
import gymnasium as gym
import gym_ligo  # Registers 'Ligo-v0'

def main():
    # --- 2. CREATE ENVIRONMENT VIA GYM ---
    # This tests that your package is installed and registered correctly.
    env = gym.make("Ligo-v0", render_mode="human")

    # --- 3. UNWRAP FOR INTERNAL ACCESS ---
    # gym.make() wraps the env in a TimeLimit wrapper.
    # To access internal variables like _last_outputs or _system,
    # we need a reference to the raw base environment.
    raw_env = env.unwrapped

    obs, info = env.reset()
    print("Environment reset.")
    print("Observation shape:", obs.shape)

    pitch_log = []
    beam_spot_log = []
    power_log = []
    actuation_log = []
    reward_log = []

    step = 0
    RENDER_EVERY_N_STEPS = 20

    while True:
        # Action must be an array of shape (2,)
        action = np.zeros(2, dtype=np.float32)

        # Step the wrapper (keeps track of time limits)
        obs, reward, terminated, truncated, info = env.step(action)

        # Only render occasionally
        if step % RENDER_EVERY_N_STEPS == 0:
            env.render()

        # --- EXTRACT DATA FOR LOGGING ---
        # We access 'raw_env' to peek at internal physics state.

        # 1. Pitch
        pitch_log.append(raw_env._last_outputs["pitch"].copy())

        # 2. Actuation
        # raw_env._current_rl_action is the exact numpy array we injected
        actuation_log.append(raw_env._current_rl_action.copy())

        # 3. Power (available in info, but also in internal state)
        power_log.append(info["cavity_power"])
        reward_log.append(reward)

        # 4. Beam Spot requires looking at the Beam component directly
        # We'll search for it dynamically on the raw_env
        if not hasattr(raw_env, '_beam_component'):
            # Cache it so we don't search every step
            raw_env._beam_component = next(
                c for c in raw_env._system.components
                if c.__class__.__name__ == 'Beam'
            )

        beam_spot_log.append(raw_env._beam_component.BS.copy())

        step += 1

        if step % 1000 == 0:
            print(f"Step {step}")

        if terminated:
            print("Terminated due to physical condition (Lock Lost).")
            break

        if truncated:
            print("Episode finished (max_steps reached).")
            break

    env.close()

    # Convert to numpy arrays
    pitch_log = np.array(pitch_log)
    beam_spot_log = np.array(beam_spot_log)
    power_log = np.array(power_log)
    actuation_log = np.array(actuation_log)
    reward_log = np.array(reward_log)

    print("\nRollout summary:")
    print("Pitch shape:", pitch_log.shape)
    print("Beam spot shape:", beam_spot_log.shape)
    print("Power shape:", power_log.shape)
    print("Actuation shape:", actuation_log.shape)
    print("Reward shape: ", reward_log.shape)

    # Basic sanity checks
    assert np.all(np.isfinite(pitch_log)), "Non-finite pitch detected"
    assert np.all(np.isfinite(beam_spot_log)), "Non-finite beam spot detected"
    assert np.all(np.isfinite(power_log)), "Non-finite power detected"

    print("\nSanity checks passed.")

    # Save logs
    np.savetxt("gym_pitch.csv", pitch_log)
    np.savetxt("gym_beam_spot.csv", beam_spot_log)
    np.savetxt("gym_power.csv", power_log)
    np.savetxt("gym_actuation.csv", actuation_log)
    np.savetxt("gym_rewards.csv", reward_log)

    print("Saved Gym rollout CSVs.")

if __name__ == "__main__":
    main()