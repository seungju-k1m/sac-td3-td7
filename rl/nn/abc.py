from abc import ABC

import torch
from torch import nn


class ACTOR(ABC, nn.Module):
    """Abstract Class for Actor."""

    def inference_mean(self, *args, **kwargs) -> torch.Tensor:
        """Inference mean."""
        pass

    def inference_mean_logvar(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference mean and log variance."""
        pass

    def inference_policy(self, *args, **kwargs) -> torch.Tensor:
        """Inference policy."""
        pass


class CRITIC(ABC, nn.Module):
    """Abstract Class for Critic."""

    def estimate_state_value(self, *args, **kwargs) -> torch.Tensor:
        """Inference mean."""
        pass

    def estimate_q_value(self, *args, **kwargs) -> torch.Tensor:
        """Inference mean and log variance."""
        pass


class ENCODER(ABC, nn.Module):
    """Abstract Class for Encoder."""

    def encode_state(self, *args, **kwargs) -> torch.Tensor:
        """Encode state."""
        pass

    def encode_state_action(self, *args, **kwargs) -> torch.Tensor:
        """Encode state."""
        pass


class WORLDMODEL(ENCODER):
    """Abstract Class for World Model."""

    def predict_reward(self, *args, **kwargs) -> torch.Tensor:
        """Predict reward."""
        pass
