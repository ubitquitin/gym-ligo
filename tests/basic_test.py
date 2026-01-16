import gymnasium as gym
import gym_ligo  # Registers 'Ligo-v0'

# Create the environment
env = gym.make("Ligo-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    # Action: [ITM_Force, ETM_Force]
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    # Render the mirrors and beam
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()