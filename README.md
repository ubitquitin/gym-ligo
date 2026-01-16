# gym-ligo: RL Environment for Gravitational Wave Detectors

`gym-ligo` is a custom [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment that wraps the **Lightsaber** time-domain simulation for LIGO interferometers. It allows researchers to train Reinforcement Learning (RL) agents to stabilize the high-power laser cavities of gravitational wave detectors.

## Features
* **Realistic Physics:** Powered by the [Janosch314/Lightsaber](https://github.com/janosch314/Lightsaber) simulation engine (O3 LIGO Hanford model).
* **Trojan Horse Control:** Surreptitiously replaces the standard PID controllers with RL actions, allowing for "Deep Loop Shaping."
* **Frequency Domain Rewards:** Custom reward functions that penalize noise in specific frequency bands (e.g., 10-30Hz).
* **Visualization:** Real-time matplotlib rendering of mirror pitch and beam spot position.

## Installation

### Prerequisites
You must have `git` installed to pull the physics engine submodule.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/yourusername/gym-ligo.git](https://github.com/yourusername/gym-ligo.git)
cd gym-ligo
```
