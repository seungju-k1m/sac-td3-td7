from typing import Any, SupportsFloat
import gymnasium as gym


class RepeatedActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """."""

    def __init__(self, env: gym.Env, repeat_action: int = 2):
        super().__init__(env)
        self._repeat_action = repeat_action

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        idx = 0
        reward = 0.0
        while idx < self._repeat_action:
            next_obs, _reward, terminated, truncated, info = self.env.step(action)
            idx += 1
            reward += _reward
            if terminated or truncated or idx == self._repeat_action:
                break
        return next_obs, reward, terminated, truncated, info


def make_env(env_id: str, **kwargs) -> gym.Env:
    """Make ENV."""
    env = gym.make(env_id, **kwargs)
    if "dm_control" in env_id:
        env = RepeatedActionWrapper(env)
        env = gym.wrappers.TimeLimit(env, 500)
        env = gym.wrappers.FlattenObservation(env)
    return env
