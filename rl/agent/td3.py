"""Soft Actor Critic."""

import json
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

import torch
import gymnasium as gym
from torch import nn

from rl import SAVE_DIR
from rl.agent.abc import Agent
from rl.replay_buffer.lap import LAPReplayBuffer
from rl.replay_buffer.simple import SimpleReplayBuffer
from rl.sampler import Sampler
from rl.neural_network import calculate_grad_norm, MLPPolicy, MLPQ
from rl.utils.annotation import ACTION, BATCH, DONE, STATE, REWARD
from rl.utils.miscellaneous import (
    convert_dict_as_param,
    get_state_action_dims,
    setup_logger,
)
from rl.runner import run_rl


class TD3(Agent, Sampler):
    """Agent for TD3."""

    def __init__(
        self,
        env_id: str,
        discount_factor: float = 0.99,
        hidden_sizes: list[int] = [256, 256],
        action_fn: str | nn.Module = "ReLU",
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        tau: float = 0.005,
        use_lap: bool = False,
        **kwargs,
    ) -> None:
        """Initialize."""
        assert env_id in gym.registry
        # Make neural network.
        state_dim, action_dim = get_state_action_dims(env_id)
        self.policy = MLPPolicy(state_dim, action_dim, hidden_sizes, action_fn)
        self.q1 = MLPQ(state_dim, action_dim, hidden_sizes, action_fn)
        self.q2 = MLPQ(state_dim, action_dim, hidden_sizes, action_fn)
        self.target_policy = deepcopy(self.policy)
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)

        # Save cconiguration.
        self.discount_factor = discount_factor
        self.target_policy_noise = target_policy_noise
        self.exploration_noise = exploration_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.use_lap = use_lap
        self.tau = tau
        self.n_runs = 0
        self.device = torch.device("cpu")
        self.to(self.device)

        # Make optimizer.
        self.make_optimizers(policy_lr, critic_lr)

    def to(self, device: torch.device) -> None:
        """Attach device."""
        self.policy = self.policy.to(device)
        self.target_policy = self.policy.to(device)
        self.q1 = self.q1.to(device)
        self.q2 = self.q2.to(device)
        self.target_q1 = self.target_q1.to(device)
        self.target_q2 = self.target_q2.to(device)
        self.device = device

    def load_state_dict(self, agent: "TD3") -> None:
        """Load state dict."""
        self.q1.load_state_dict(agent.q1.state_dict())
        self.q2.load_state_dict(agent.q2.state_dict())
        self.policy.load_state_dict(agent.policy.state_dict())
        self.target_q1.load_state_dict(agent.target_q1.state_dict())
        self.target_q2.load_state_dict(agent.target_q2.state_dict())

    def make_optimizers(self, policy_lr: float, critic_lr: float) -> None:
        """Make optimizers."""
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )

    def zero_grad(self) -> None:
        """Apply zero gradient."""
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()

    def sample(self, state: STATE, deterministic: bool = False, **kwargs) -> ACTION:
        """Sample action."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.Tensor(state).float()
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            action = self._inference_action(state)
            action = action.cpu().detach().numpy()[0]
            if not deterministic:
                noise = np.random.normal(
                    0, 1.0 * self.exploration_noise, size=action.shape
                )
                action += noise
                action = np.clip(action, -1.0, 1.0)
        return action

    def _inference_action(self, state: STATE) -> ACTION:
        """Only forward with policy."""
        mean = self.policy.forward(state)
        action = torch.tanh(mean)
        return action

    @staticmethod
    def _lap_huber(td_error: torch.Tensor, min_priority: float = 1.0) -> torch.Tensor:
        """ "Caluclate Huber loss for LAP."""
        return torch.where(
            td_error < min_priority, 0.5 * td_error.pow(2), min_priority * td_error
        ).mean()

    def _q_train_ops(
        self,
        state: STATE,
        action: REWARD,
        next_state: STATE,
        reward: REWARD,
        done: DONE,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # make q target
        with torch.no_grad():
            noise = (torch.rand_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (torch.tanh(self.target_policy(next_state)) + noise).clamp(
                -1.0, 1.0
            )
            next_value = torch.min(
                self.target_q1.forward(next_state, next_action),
                self.target_q2.forward(next_state, next_action),
            )
            q_target = reward + self.discount_factor * next_value * done
        # calculate q value
        q1, q2 = self.q1.forward(state, action), self.q2.forward(state, action)
        if self.use_lap:
            td_loss1 = (q1 - q_target).abs()
            td_loss2 = (q2 - q_target).abs()
            q1_loss = self._lap_huber(td_loss1)
            q2_loss = self._lap_huber(td_loss2)
            q_loss = q1_loss + q2_loss
            # TODO: It is hard-coding
            priority = torch.max(td_loss1, td_loss2).clamp(1.0).pow(0.4).view(-1)
            return q_loss, priority.cpu().detach()
        else:
            q1_loss = torch.mean((q_target - q1) ** 2.0) * 0.5
            q2_loss = torch.mean((q_target - q2) ** 2.0) * 0.5
            q_loss = q1_loss + q2_loss
            return q_loss

    def _policy_train_ops(self, state: STATE, **kwargs) -> torch.Tensor:
        """Policy ops."""
        # Calculate policy loss.
        action = self._inference_action(state)
        q1, q2 = self.q1.forward(state, action), self.q2.forward(state, action)
        policy_loss = -torch.min(q1, q2).mean()
        return policy_loss

    @torch.no_grad()
    def _update_target_q_fns(self) -> None:
        """Update target value function."""
        for q_parm, t_q_parm in zip(self.q1.parameters(), self.target_q1.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))
        for q_parm, t_q_parm in zip(self.q2.parameters(), self.target_q2.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))
        for pi_parm, t_pi_parm in zip(
            self.policy.parameters(), self.target_policy.parameters()
        ):
            t_pi_parm.copy_(self.tau * pi_parm + t_pi_parm * (1 - self.tau))

    def train_ops(
        self,
        batch: BATCH,
        replay_buffer: LAPReplayBuffer | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Run train ops."""
        info = {}
        batch = {key: value.to(self.device) for key, value in batch.items()}

        # Update Action-State value function.
        q_value_loss = self._q_train_ops(**batch)
        if isinstance(q_value_loss, tuple):
            q_value_loss, priority = q_value_loss
            assert isinstance(replay_buffer, LAPReplayBuffer)
            replay_buffer.update_priority(priority)
        q_value_loss.backward()
        info["norm/q1_value"] = calculate_grad_norm(self.q1)
        info["norm/q2_value"] = calculate_grad_norm(self.q2)
        self.optim_q_fns.step()
        self.zero_grad()
        info["train/q_fn"] = float(q_value_loss.cpu().detach().numpy())

        # Update policy.
        info["train/policy"] = None
        info["norm/policy"] = None
        if self.n_runs % self.policy_freq == 0:
            policy_loss = self._policy_train_ops(**batch)
            policy_loss.backward()
            info["train/policy"] = float(policy_loss.cpu().detach().numpy())
            info["norm/policy"] = calculate_grad_norm(self.policy)

            self.optim_policy.step()
            self.zero_grad()
            # Update Target
            self._update_target_q_fns()
        self.n_runs += 1
        return info


def run_td3(
    rl_run_name: str,
    env_id: str,
    seed: int = 777,
    use_lap: bool = False,
    replay_buffer_size: int = 1_000_000,
    benchmark_idx: int = 0,
    record_video: bool = False,
    show_progressbar: bool = True,
    **kwargs,
) -> None:
    """Run Heating Environment."""
    params = convert_dict_as_param(deepcopy(locals()))
    params["rl_alg"] = "TD3"
    print("-" * 5 + "[TD3]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    if benchmark_idx > 0:
        base_dir = (
            SAVE_DIR
            / "VALID"
            / env_id
            / "TD3"
            / f"TD3-{rl_run_name}"
            / str(benchmark_idx)
        )
        show_progressbar = False
        record_video = False
    else:
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
        rl_run_name = f"{rl_run_name}-{timestamp}"
        base_dir = SAVE_DIR / "TD3" / rl_run_name
    base_dir.mkdir(exist_ok=True, parents=True)

    # TODO: Replace logger by mlflow.
    logger = setup_logger(str(base_dir / "training.log"))

    # Write out configuration file.
    with open(base_dir / "config.json", "w") as file_handler:
        json.dump(params, file_handler, indent=4)
    # Set Seed.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Make envs
    env = gym.make(env_id)
    agent = TD3(
        env_id,
        use_lap=use_lap,
        **kwargs,
    )
    replay_buffer = (
        SimpleReplayBuffer(replay_buffer_size)
        if not use_lap
        else LAPReplayBuffer(
            replay_buffer_size, env.observation_space, env.action_space
        )
    )
    run_rl(
        env,
        agent,
        replay_buffer,
        logger,
        show_progressbar=show_progressbar,
        record_video=record_video,
        **kwargs,
    )
