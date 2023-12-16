RL_HELP = """
Run Off-Policy Reinforcement Learning Algorithms.


Examples:

    # I recommend reading docstring of each command.

    >>> rl sac --help

    # Run default setting

    >>> rl sac

    # You can load hyper-parameter via *.yaml

    >> rl sac -c config/common.yaml

    # You can specify the run name.

    >>> rl sac --run-name Ant@Seed42

    # You want to record the video while training,

    >>> rl sac --record-video
"""

SAC_HELP = """
Soft Actor Critic (SAC) Algorithm.

Here is paper address:
https://arxiv.org/abs/1801.01290

Notes:

    - SAC is one of the most famous off-policy
reinforcement learning algorithm. It is often used as baseline.

    - One of the key hyper-parameter for SAC is temperature, `tmp`.
If you set the `tmp` as negative, auto-tmp mode works, which automatically
tuning the temperature during training.

Examples:

    # If you want to run SAC alg with fixed temperature.\n
    >>> python cli.py rl sac --env-id Ant-v4 --tmp 0.2 --n-iteration 1_000_000\n
    # If you want to record video while training.\n
    >>> python cli.py rl sac --env-id Ant-v4 --record-video
"""

TD3_HELP = """
Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3).

Here is paper address:
https://arxiv.org/pdf/1802.09477.pdf

Notes:

    - TD3 is a family of DDPG style algorithms.
This algorithm handles the overestimation of value function introducing
two state-action value functions.

Examples:

    # If you want to run TD3 Algorithm for Ant-v4 of mujoco env with `td3@
ant` run name.\n
    >>> python cli.py rl td3 --env-id Ant-v4 --run-name td3@ant\n

"""

TD7_HELP = """
TD7 is variant of TD3 algorithm with 4 additions.

Here is paper address:
https://arxiv.org/pdf/2306.02451.pdf

Notes:

    - State-Action Learned Embedding (SALE) Network.

    - LAP Replay Memory.

    - Checkpoint Policy Strategy.

    - Behavior Cloning Term for offline RL.


Examples:

    >>> python cli.py rl td7 --env-id Ant-v4 --run-name td7@ant\n

"""
