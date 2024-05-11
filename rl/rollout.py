"""Rollout."""
from copy import deepcopy
from typing import Any

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from rl.replay_memory.base import REPLAYMEMORY


from rl.sampler import SAMPLER, RandomSampler
from rl.utils.annotation import STATE, BATCH


class Rollout:
    """Rollout Worker."""

    def __init__(
        self,
        env: RecordEpisodeStatistics,
        replay_buffer: REPLAYMEMORY,
        rollout_kwargs: None | dict[str, Any] = None,
    ):
        """Initialize."""
        self.env = env
        self.replay_buffer = replay_buffer
        self.count_replay_buffer: int = 0
        self.sampler = RandomSampler(self.env.action_space)
        self.need_reset: bool = True
        self.n_episode: int = 0
        self.obs: STATE
        self.rollout_kwargs = rollout_kwargs or {}
        self._is_first_obs = False

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
            self.obs = self.env.reset()[0]
            self._is_first_obs = True
        action = self.sampler.sample(
            self.obs, **self.rollout_kwargs, is_first_obs=self._is_first_obs
        )
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated
        self.replay_buffer.append(
            deepcopy([self.obs, action, reward, next_obs, 1.0 - float(terminated)])
        )
        self.obs = next_obs
        self.need_reset = done
        if self._is_first_obs:
            self._is_first_obs = False
        return done
