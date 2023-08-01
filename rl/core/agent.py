from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
from typing import TypeVar


from rl.utils.annotation import BATCH, PATH

AGENT = TypeVar("AGENT", bound="Agent")


class Agent(ABC):
    """Base Agent."""

    @abstractmethod
    def make_nn(self) -> None:
        """Build neural network."""

    @abstractmethod
    def make_optimizers(self) -> None:
        """Make Optimizer."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Apply zero grad with respect to optimizers."""

    @abstractmethod
    def step(self):
        """Apply gradient-descent algorithm."""

    def train_ops(self, batch: BATCH, *args, **kwargs) -> None:
        """Train ops.

        It corresponds to the one iteration with given batch.
        """
        raise NotImplementedError("`train_ops` should be implemented.")

    def save(self, path: PATH) -> None:
        """Save the algorithm."""
        path = Path(path) if isinstance(path, str) else path
        dir = path.parent
        os.makedirs(dir, exist_ok=True)
        with open(path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    @staticmethod
    def load(path: PATH) -> AGENT:
        """Load the algorithm."""
        assert os.path.isfile(path)
        with open(path, "rb") as file_handler:
            self: AGENT = pickle.load(file_handler)
        return self
