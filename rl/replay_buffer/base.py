from abc import ABC
from typing import Any, TypeVar
from rl.utils.annotation import BATCH


REPLAYBUFFER = TypeVar("REPLAYBUFFER", bound="BaseReplayBuffer")


class BaseReplayBuffer(ABC):
    def __init__(self, replay_buffer_size: int, **kwargs) -> None:
        """Initialize."""
        self.replay_buffer_size = replay_buffer_size

    def sample(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Sample Batch."""
        raise NotImplementedError("Sample method should be implemented.")

    def append(self, transition: list[Any]) -> None:
        """Append."""
        raise NotImplementedError("Append method should bed implemented.")

    def __len__(self) -> int:
        """Return length."""
        raise NotImplementedError("__len__ should be implemented.")
