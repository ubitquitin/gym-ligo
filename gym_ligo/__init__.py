import sys
import os
from gymnasium.envs.registration import register

# --- 1. FIND THE SUBMODULE ---
# Get the directory where this __init__.py lives
package_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to root, then down into extern/Lightsaber/Lightsaber_X
# (We point to Lightsaber_X because that's where simulate.py lives)
submodule_path = os.path.join(package_dir, "..", "extern", "Lightsaber", "Lightsaber_X")
submodule_path = os.path.normpath(submodule_path)

# --- 2. ADD TO PYTHON PATH ---
if submodule_path not in sys.path:
    print(f"gym-ligo: Adding {submodule_path} to sys.path")
    sys.path.append(submodule_path)

# --- 3. REGISTER ENV ---
register(
    id="Ligo-v0",
    entry_point="gym_ligo.envs:LightsaberEnv",
    max_episode_steps=262143, # Remember the off-by-one fix!
)