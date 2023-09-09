[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)

# Off-policy Reinforcement Learning Algorithms with Know-how.

Research into the advancement of reinforcement learning algorithms and their integration with deep learning has been actively conducted, leading to significant progress in deep reinforcement learning in recent times. In particular, attempts to apply deep learning to reinforcement learning have been facilitated by the recent developments in deep learning, resulting in many papers addressing this topic. 

However, since deep learning is an empirically-driven field, using only established facts can be highly inefficient. Methods used to boost performance but not yet proven are referred to as "know-how." These know-how techniques are openly used and play a crucial role. 

**In this repository, various know-how techniques that have been used in deep reinforcement learning have been implemented as clean code. Also we have benchmarked using the Mujoco benchmark with at least 8 seeds.**

---
## Installation

```bash
# Clone repo.
git clone https://github.com/seungju-k1m/reinforcement-learning

# Change Directory.
cd reinforcement-learning

# Make-up virtual environment. (python version is 3.10)
python -m venv .venv --prompt rl

# Activate virtual env.
source .venv/bin/activate

# install
pip install -r requirements.txt

```



----
## Off-Policy Algorithm


We have chosen two of the most commonly used off-policy reinforcement learning algorithms as representatives. SAC and TD3 are examples of these algorithms.

1. [SAC](https://github.com/haarnoja/sac)
2. [TD3](https://github.com/sfujim/TD3)

---
## Know-Hows