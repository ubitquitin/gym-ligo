import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from collections import deque
try:
    from simulate import create_system, link_components # Works because of the path hack
except ImportError as e:
    raise ImportError(
        "Could not import 'simulate'. "
        "Did you initialize submodules? Run: git submodule update --init --recursive"
    ) from e

# --- CONFIGURATION ---
HISTORY_LEN = 64  # How many past steps the agent sees
ACT_SCALE = 1e-5  # Scaling factor: Agent [-1, 1] -> Physics [Newtons/Torque]
TARGET_BAND = (10.0, 30.0) # Critical frequency band to suppress (Hz)
metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

class LightsaberEnv(gym.Env):
    """
    Gymnasium wrapper for LIGO Lightsaber simulation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path, render_mode=None):
        super().__init__()
        self.config_path = config_path

        # --- Define Action Space ---
        # 2 Continuous actions: [ITM_Control, ETM_Control]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Define Observation Space ---
        # Flattened history of sensor readouts
        obs_dim = 2 * HISTORY_LEN
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Internal State ---
        self._system = None
        self._controller_ref = None
        self._step_count = 0
        self.max_steps = None
        self._dt = None

        # Initialize the action cache to zeros
        self._current_rl_action = np.zeros(2, dtype=np.float32)

        # History buffer
        self._obs_buffer = deque(maxlen=HISTORY_LEN)

        # Rendering stuff
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._current_rl_action = np.zeros(2, dtype=np.float32)

        # 1. Clear History
        self._obs_buffer.clear()
        for _ in range(HISTORY_LEN):
            self._obs_buffer.append(np.zeros(2, dtype=np.float32))

        # 2. Build System
        self._system, self._simulation, self._output = self._build_lightsaber_system()

        # 3. Step once to prime the system
        # (We send zero action for the very first priming step)
        self._inputs = self._get_initial_inputs()
        self._last_outputs = self._system.step(self._inputs)
        self._inputs = self._last_outputs.copy()

        # 4. Return initial observation
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        self._step_count += 1

        # 1. APPLY ACTION (The Trojan Horse)
        real_action = self._apply_action(action)

        # 2. STEP PHYSICS
        self._step_lightsaber()

        # 3. OBSERVE
        obs = self._get_observation()

        # 4. REWARD
        reward = self._compute_reward(obs, real_action)

        # 5. TERMINATION
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_steps

        info = {
            "cavity_power": self._last_outputs.get("cavity_power", 0.0),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Core Logic
    # ------------------------------------------------------------------

    def _build_lightsaber_system(self):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Separate sections
        simulation = next(v for k, v in config.items() if v.get("type") == "Simulation")
        output = next(v for k, v in config.items() if v.get("type") == "Output")

        config_clean = {k: v for k, v in config.items()
                        if v.get("type") not in ["Simulation", "Output"]}

        # Create System
        system = create_system(
            config=config_clean,
            simulation=simulation,
            plot_dir=output["out_directory"],
        )

        link_components(system)

        # --- FIND THE CONTROLLER TO HIJACK ---
        self._controller_ref = next(
            (c for c in system.components if hasattr(c, 'name') and c.name == 'PITCH_CONTROL'),
            None
        )

        if self._controller_ref is None:
             self._controller_ref = next(
                (c for c in system.components if c.__class__.__name__ == 'Controller'),
                None
            )

        # --- LOBOTOMIZE THE CONTROLLER ---
        if self._controller_ref:
            print("RL Env: PITCH_CONTROL found. Disabling internal logic.")
            # Use *args to swallow inputs (Fixes TypeError)
            self._controller_ref.step = lambda *args: None

            # Ensure output buffers exist
            if not hasattr(self._controller_ref, 'out'):
                 self._controller_ref.out = np.zeros(2)
            self._controller_ref.output = np.zeros(2)

        # Time params
        fs = float(simulation["simulation_sampling_frequency"])
        duration = float(simulation["duration_batch"])

        # Subtract 1 to prevent Index Error
        self.max_steps = int(duration * fs) - 1
        self._dt = 1.0 / fs

        return system, simulation, output

    def _get_initial_inputs(self):
        return {
            "pitch": np.zeros((2,)),
            "in_power": 0.0,
            "readout": np.zeros((2,)),
            "rad_torque": np.zeros((2,)),
            "act_mirror": np.zeros((2,)),
            "act_sus": np.zeros((2,)),
        }

    def _step_lightsaber(self):
        outputs = self._system.step(self._inputs)
        self._last_outputs = outputs
        self._inputs = outputs.copy()

    def _apply_action(self, action):
        # Scale Action
        scaled_action = np.clip(action, -1.0, 1.0) * ACT_SCALE

        # Store locally (Fixes the AttributeError)
        self._current_rl_action = scaled_action

        # INJECTION
        if self._controller_ref is not None:
            # FORCE the variable used in Lightsaber.py
            self._controller_ref.out = scaled_action
            self._controller_ref.output = scaled_action

        # Redundancy
        self._inputs["act_sus"] = scaled_action

        return scaled_action

    def _get_observation(self):
        latest_readout = self._last_outputs["readout"]
        self._obs_buffer.append(latest_readout)
        flat_obs = np.array(self._obs_buffer, dtype=np.float32).flatten()
        return flat_obs

    def _compute_reward(self, obs, real_action):
        history = np.array(self._obs_buffer)

        # --- 1. RMS Error (The biggest change) ---
        # Assume typical error is 1e-10. Squared is 1e-20.
        # We want that to be a penalty of -1.0.
        # So we need a multiplier of 1e19 or 1e20.
        rms_error = np.mean(history[-1]**2)
        r_rms = -1e20 * rms_error  # Changed from -10.0

        # --- 2. Control Effort ---
        # Real action is ~1e-5. Squared is 1e-10.
        # We want a penalty of roughly -0.1 for max effort.
        # So we need a multiplier of 1e9.
        control_effort = np.sum(real_action**2)
        r_ctrl = -1e9 * control_effort # Changed from -0.1

        # --- 3. Frequency Domain ---
        # This is also based on squared errors, so it needs the huge multiplier.
        if len(history) >= 16:
            soft_mode_trace = history[:, 0]
            fft_vals = np.fft.rfft(soft_mode_trace)
            fft_freq = np.fft.rfftfreq(len(soft_mode_trace), d=self._dt)

            mask = (fft_freq >= TARGET_BAND[0]) & (fft_freq <= TARGET_BAND[1])
            band_power = np.sum(np.abs(fft_vals[mask])**2)

            # Penalize specific band noise heavily
            r_freq = -1e21 * band_power # Changed from -100.0
        else:
            r_freq = 0.0

        return r_rms + r_ctrl + r_freq

    def _check_termination(self):
        readout = self._last_outputs["readout"]
        if not np.all(np.isfinite(readout)):
            return True
        if np.max(np.abs(readout)) > 1.0:
            return True
        return False


    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt

        # 1. Initialize Plot (Run once)
        if self.fig is None:
            plt.ion() # Interactive mode
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
            self.fig.suptitle("LIGO Cavity State")

            # Left Plot: Mirror Pitch (Side View)
            self.ax[0].set_title("Mirror Pitch (microradians)")
            self.ax[0].set_ylim(-2, 2)
            self.ax[0].set_xlim(-1, 2)
            # Create "Mirror" bars
            self.itm_line, = self.ax[0].plot([0, 0], [-1, 1], 'b-', lw=5, label="ITM")
            self.etm_line, = self.ax[0].plot([1, 1], [-1, 1], 'r-', lw=5, label="ETM")
            self.beam_line, = self.ax[0].plot([0, 1], [0, 0], 'g--', alpha=0.5)
            self.ax[0].legend()

            # Right Plot: Beam Spot on ETM (Front View)
            self.ax[1].set_title("Beam Spot on ETM (mm)")
            self.ax[1].set_xlim(-5, 5)
            self.ax[1].set_ylim(-5, 5)
            self.ax[1].add_artist(plt.Circle((0, 0), 2.0, color='r', fill=False, label="Mirror Edge"))
            self.spot_dot, = self.ax[1].plot([], [], 'go', ms=10, label="Laser Spot")
            self.ax[1].grid(True)

        # 2. Update Data
        # Get latest data (convert to microradians/millimeters for readability)
        pitch = self._last_outputs["pitch"] * 1e6
        beam_spot = self._system.components[2].BS * 1e3 # Assuming component 2 is Beam

        # Update ITM (Blue) - simple rotation visualization
        self.itm_line.set_xdata([0 - pitch[0], 0 + pitch[0]]) # Exaggerated tilt

        # Update ETM (Red)
        self.etm_line.set_xdata([1 - pitch[1], 1 + pitch[1]])

        # Update Beam Line (Green)
        #self.beam_line.set_ydata([0, (pitch[0] - pitch[1])]) # Rough ray trace visualization
        self.beam_line.set_ydata([0, pitch[0]])

        # Update Spot
        self.spot_dot.set_data([beam_spot[0]], [beam_spot[1]])

        # 3. Draw
        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.00001)
        elif self.render_mode == "rgb_array":
            # Convert canvas to array for video recording wrappers
            self.fig.canvas.draw()
            return np.array(self.fig.canvas.renderer.buffer_rgba())