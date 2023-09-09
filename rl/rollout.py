"""Rollout."""
from copy import deepcopy

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
from rl.replay_buffer.base import REPLAYBUFFER


from rl.sampler import SAMPLER, RandomSampler
from rl.utils.annotation import OBSERVATION, BATCH


class Rollout:
    """Rollout Worker."""

    def __init__(
        self,
        env: RecordEpisodeStatistics,
        replay_buffer: REPLAYBUFFER,
    ):
        """Initialize."""
        self.env = env
        self.replay_buffer = replay_buffer
        self.count_replay_buffer: int = 0
        self.sampler = RandomSampler(self.env.action_space)
        self.need_reset: bool = True
        self.n_episode: int = 0
        self.obs: OBSERVATION

    def set_sampler(self, sampler: SAMPLER) -> None:
        """Set sampler."""
        self.sampler = sampler

    def get_batch(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Return batch for train ops."""
        return self.replay_buffer.sample(batch_size, use_torch)

    def sample(self) -> bool:
        """Sample action using policy."""
        if self.need_reset:
            self.need_reset = False
            self.obs = self.env.reset(seed=np.random.randint(1, 1000000))[0]
        action = self.sampler.sample(self.obs)
        next_obs, reward, truncated, terminated, info = self.env.step(action)
        done = truncated or terminated
        self.replay_buffer.append(
            deepcopy([self.obs, action, reward, next_obs, 1.0 - float(done)])
        )
        self.obs = next_obs
        self.need_reset = done
        return done
