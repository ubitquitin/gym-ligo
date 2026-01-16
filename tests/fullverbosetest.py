import numpy as np
from gym_ligo.LightsaberEnv import LightsaberEnv

def main():
    config_path = "configuration/config.yaml"
    env = LightsaberEnv(config_path, render_mode="human")

    obs, info = env.reset()
    print("Environment reset.")
    print("Observation shape:", obs.shape)  # Changed from keys() to shape

    pitch_log = []
    beam_spot_log = []
    power_log = []
    actuation_log = []
    reward_log = []

    step = 0

    RENDER_EVERY_N_STEPS = 20

    while True:
        # Action must be an array of shape (2,), not a scalar 0
        action = np.zeros(2, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # Only render occasionally
        if step % RENDER_EVERY_N_STEPS == 0:
            env.render()

        # --- EXTRACT DATA FOR LOGGING ---
        # The 'obs' variable is now just the history for the Neural Net.
        # To log physics, we peek into the environment's internal state.

        # 1. Pitch is reliable
        pitch_log.append(env._last_outputs["pitch"].copy())

        # 2. ACTUATION FIX: Read from the cached RL action, not the output dict
        # env._current_rl_action is the exact numpy array we injected
        actuation_log.append(env._current_rl_action.copy())

        # 3. Power from info
        power_log.append(info["cavity_power"])
        reward_log.append(reward)

        # 3. Beam Spot requires looking at the Beam component directly
        # We find the beam component in the system (usually index 2, or searchable)
        # We'll search for it dynamically to be safe:
        if not hasattr(env, '_beam_component'):
            # Cache it so we don't search every step
            env._beam_component = next(c for c in env._system.components if c.__class__.__name__ == 'Beam')

        beam_spot_log.append(env._beam_component.BS.copy())

        step += 1

        if step % 1000 == 0:
            print(f"Step {step}")

        if terminated:
            print("Terminated due to physical condition (Lock Lost).")
            break

        if truncated:
            print("Episode finished (max_steps reached).")
            break

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