"""ABC for Agent."""

import os
import pickle
from pathlib import Path
from typing import TypeVar
from abc import ABC, abstractmethod

import torch

from rl.utils.annotation import BATCH, PATH

AGENT = TypeVar("AGENT", bound="Agent")


class Agent(ABC):
    """Base Agent."""

    @abstractmethod
    def make_optimizers(self) -> None:
        """Make Optimizer."""

    def train_ops(self, batch: BATCH, *args, **kwargs) -> None:
        """Train ops.

        It corresponds to the one iteration with given batch.
        """
        raise NotImplementedError("`train_ops` should be implemented.")

    def to(self, device: torch.device) -> None:
        """Attach device to neural network."""
        raise NotImplementedError("`to` should be implemented.")

    def load_state_dict(self, agent: "Agent") -> None:
        """Attach device to neural network."""
        raise NotImplementedError("`load_state_dict` should be implemented.")

    def save(self, path: PATH) -> None:
        """Save the algorithm."""
        current_device = self.device
        self = self.to(torch.device("cpu"))
        path = Path(path) if isinstance(path, str) else path
        dir = path.parent
        os.makedirs(dir, exist_ok=True)
        with open(path, "wb") as file_handler:
            pickle.dump(self, file_handler)
        self = self.to(current_device)

    @staticmethod
    def load(path: PATH) -> AGENT:
        """Load the algorithm."""
        assert os.path.isfile(path)
        with open(path, "rb") as file_handler:
            self: AGENT = pickle.load(file_handler)
        return self
