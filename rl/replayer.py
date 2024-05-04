import json
from pathlib import Path
from pprint import pprint
import random
from typing import Any
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
import torch
import yaml

from rl.agent.abc import AGENT
from rl import agent as rl_agent
from rl.utils import NoStdStreams, make_env


class Replayer:
    """Replayer."""

    def __init__(
        self,
        root_dir: str | Path,
        use_ckpt_model: bool = False,
        seed: int = 42,
        video_dir: str | Path | None = None,
    ) -> None:
        """Initialize."""
        # Set path.
        root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir

        # Set which agent.
        weight_name = "ckpt.pkl" if use_ckpt_model else "best.pkl"

        # Load configuration file.
        if (root_dir / "config.yaml").is_file():
            with open(root_dir / "config.yaml") as file_handler:
                config: dict[str, Any] = yaml.load(file_handler, yaml.FullLoader)
        else:
            with open(root_dir / "config.json") as file_handler:
                config: dict[str, Any] = json.load(file_handler)

        # Load Agent.
        rl_alg: str = config["rl_alg"]
        agent: AGENT = getattr(rl_agent, rl_alg)
        self.agent = agent.load(root_dir / weight_name)

        # Prepare video folder.
        if video_dir is None:
            video_dir = root_dir / "replayer"
        elif isinstance(video_dir, str):
            video_dir = Path(video_dir)
        video_dir.mkdir(exist_ok=True, parents=True)

        # Make env,
        env_id: str = config["env_id"]
        if "dm_control" in env_id:
            env = make_env(
                env_id, render_mode="rgb_array", render_kwargs=dict(camera_id=0)
            )
        else:
            env = make_env(env_id, render_mode="rgb_array")

        self.env = RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: True,
        )

        # Set seed.
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        pprint(config, indent=4)

    def run(self, n_episodes: int = 8, stochastic: bool = False) -> None:
        """Run."""
        env = RecordEpisodeStatistics(self.env, n_episodes)
        with NoStdStreams():
            for _ in range(n_episodes):
                obs = env.reset()[0]
                done = False
                while done is False:
                    action = self.agent.sample(obs, not stochastic)
                    next_obs, _, truncated, terminated, _ = env.step(action)
                    if truncated or terminated:
                        done = True
                    obs = next_obs
            episode_returns = np.stack(env.return_queue)
            epi_returns_mean = episode_returns.mean()
            epi_returns_min = abs(min(episode_returns) - epi_returns_mean)
            epi_returns_max = abs(max(episode_returns) - epi_returns_mean)
            epi_std = float(max(epi_returns_min, epi_returns_max))
        print(f"PERF/MEAN: {epi_returns_mean:.3f} +- {epi_std:.3f}")
