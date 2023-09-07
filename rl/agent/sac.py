"""Soft Actor Critic."""

import json
import pandas as pd
from copy import deepcopy
from typing import Literal
from datetime import datetime

import torch
import gymnasium as gym
from torch import nn

from rl import SAVE_DIR
from rl.agent.base import Agent
from rl.replay_buffer.base import REPLAYBUFFER
from rl.replay_buffer.lap import LAPReplayBuffer
from rl.replay_buffer.simple import SimpleReplayBuffer
from rl.sampler import Sampler
from rl.neural_network import (
    calculate_grad_norm,
    clip_grad_norm,
    make_feedforward,
)
from rl.utils.annotation import ACTION, BATCH, DONE, EPS, OBSERVATION, REWARD
from rl.utils.miscellaneous import convert_dict_as_param, setup_logger
from rl.runner import run_rl, run_rl_w_checkpoint


class SAC(Agent, Sampler):
    """Agent for Soft Actor Critic."""

    def __init__(
        self,
        action_space: gym.spaces.Dict,
        obs_space: gym.spaces.Box,
        discount_factor: float = 0.99,
        hidden_sizes: list[int] = [256, 256],
        action_fn: str | nn.Module = "ReLU",
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        step_per_batch: int = 1,
        tau: float = 0.005,
        tmp: float | Literal["auto"] = "auto",
        policy_reg_coeff: float = 0.0,
        policy_sto_reg_coeff: float = 0.0,
        reward_scale: float = 1.0,
        use_lap: bool = False,
        **kwargs,
    ) -> None:
        self.discount_factor = discount_factor
        self.hidden_sizes = hidden_sizes
        self.action_fn = action_fn
        self.obs_space = obs_space
        self.action_space = action_space
        self.action_dim = action_space.shape[-1]
        self.auto_tmp_mode = True if tmp == "auto" else False
        self.tmp = (
            nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            if self.auto_tmp_mode
            else tmp
        )
        self.make_nn()
        self.make_optimizers(policy_lr, critic_lr)
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.step_per_batch = step_per_batch
        self.tau = tau
        self.policy_reg_coeff = policy_reg_coeff
        self.policy_sto_reg_coeff = policy_sto_reg_coeff
        self.n_runs = 0
        self.reward_scale = reward_scale
        self.use_lap = use_lap

    def make_nn(self) -> None:
        """Make neurla networks."""
        obs_dim = self.obs_space.shape[-1]
        self.policy = make_feedforward(
            obs_dim, self.action_dim * 2, self.hidden_sizes, self.action_fn
        )

        self.q1 = make_feedforward(
            obs_dim + self.action_dim, 1, self.hidden_sizes, self.action_fn
        )
        self.q2 = make_feedforward(
            obs_dim + self.action_dim, 1, self.hidden_sizes, self.action_fn
        )
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def make_optimizers(self, policy_lr: float, critic_lr: float) -> None:
        """Make optimizers."""
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        if self.auto_tmp_mode:
            self.optim_tmp = torch.optim.Adam([self.tmp], lr=policy_lr)

    def zero_grad(self) -> None:
        """Apply zero gradient."""
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        if self.auto_tmp_mode:
            self.optim_tmp.zero_grad()

    def step(self) -> None:
        """Step backpropgagtion via optimizers."""
        self.optim_q_fns.step()
        self.optim_policy.step()
        if self.auto_tmp_mode:
            self.optim_tmp.step()

    def policy_forward(
        self, obs: OBSERVATION
    ) -> tuple[ACTION, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Only forward with policy."""
        mean, log_std = self.policy.forward(obs).split(self.action_dim, -1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        action_sample = distribution.rsample()
        action = torch.tanh(action_sample)
        log_pi = distribution.log_prob(action_sample).sum(-1, keepdim=True) - torch.log(
            1 - action**2.0 + EPS
        ).sum(-1, keepdim=True)
        return action, log_pi, (mean, log_std)

    def _q_value_ops(
        self,
        obs: OBSERVATION,
        action: REWARD,
        next_obs: OBSERVATION,
        reward: REWARD,
        done: DONE,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # make q target
        with torch.no_grad():
            next_action, next_log_pi = self.policy_forward(next_obs)[:2]
            next_obs_action = torch.cat((next_obs, next_action), -1)
            next_value = torch.min(
                self.target_q1.forward(next_obs_action),
                self.target_q2.forward(next_obs_action),
            )
            tmp = self.tmp.exp() if self.auto_tmp_mode else self.tmp
            q_target = (
                reward * self.reward_scale
                + self.discount_factor * (next_value - tmp * next_log_pi) * done
            )
        # calculate q value
        obs_action = torch.cat((obs, action), 1)
        q1, q2 = self.q1.forward(obs_action), self.q2.forward(obs_action)
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

    def _policy_ops(
        self, obs: OBSERVATION, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """Policy ops."""
        # Calculate policy loss.
        action, log_pi, (mean, _) = self.policy_forward(obs)
        obs_action = torch.cat((obs, action), -1)
        q1, q2 = self.q1.forward(obs_action), self.q2.forward(obs_action)
        q_value = torch.min(q1, q2)
        tmp = self.tmp.exp().detach() if self.auto_tmp_mode else self.tmp
        policy_loss = torch.mean(-q_value + log_pi * tmp)

        # Calcuate tmp loss.
        target_entropy = self.action_dim

        tmp_loss = (
            torch.mean(self.tmp.exp() * (-log_pi.detach() + target_entropy))
            if self.auto_tmp_mode
            else 0.0
        )

        # Entropy for logging.
        entropy = -(log_pi.mean().detach())

        # Regularizer for policy and logging.
        policy_reg = 0.5 * (torch.mean(mean**2))
        return policy_loss, tmp_loss, policy_reg, entropy

    def _policy_stochastic_reg_ops(
        self, obs: OBSERVATION, next_obs: OBSERVATION, *args, **kwargs
    ) -> torch.Tensor:
        """Policy Regular ops."""
        next_mean, _ = self.policy_forward(next_obs)[-1]
        mean, _ = self.policy_forward(obs)[-1]
        policy_reg = torch.mean((next_mean - mean) ** 2.0)
        return policy_reg

    def sample(self, obs: OBSERVATION, deterministic: bool = False, **kwargs) -> ACTION:
        """Sample action for inference action."""
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.Tensor(obs).float()
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            if deterministic:
                mean, _ = self.policy_forward(obs)[-1]
                action = torch.tanh(mean)
            else:
                action = self.policy_forward(obs)[0]
            action = action.cpu().detach().numpy()[0]
        return action

    @torch.no_grad()
    def update_target_value_fn(self) -> None:
        """Update target value function."""
        for q_parm, t_q_parm in zip(self.q1.parameters(), self.target_q1.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))
        for q_parm, t_q_parm in zip(self.q2.parameters(), self.target_q2.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))

    def train_ops(
        self, batch: BATCH, replay_buffer: REPLAYBUFFER | None = None, *args, **kwargs
    ) -> None:
        """Run train ops."""
        # Update state value function.
        info = {}
        max_norm = float("inf")
        # Update Action-State value function.
        q_value_loss = self._q_value_ops(**batch)
        if isinstance(q_value_loss, tuple):
            q_value_loss, priority = q_value_loss
        q_value_loss.backward()
        info["norm/q1_value"] = calculate_grad_norm(self.q1)
        info["norm/q2_value"] = calculate_grad_norm(self.q2)
        clip_grad_norm(self.q1, max_norm)
        clip_grad_norm(self.q2, max_norm)
        self.optim_q_fns.step()
        self.zero_grad()
        info["loss/q_value"] = float(q_value_loss.cpu().detach().numpy())

        # Update policy.
        policy_loss, tmp_loss, policy_reg, entropy = self._policy_ops(**batch)
        policy_stochastic_reg = self._policy_stochastic_reg_ops(**batch)
        loss = (
            policy_loss
            + self.policy_reg_coeff * policy_reg
            + self.policy_sto_reg_coeff * policy_stochastic_reg
        )
        if self.auto_tmp_mode:
            loss += tmp_loss
        loss.backward()
        info["norm/policy"] = calculate_grad_norm(self.policy)
        clip_grad_norm(self.policy, max_norm)

        # Update temperatgure.
        if self.auto_tmp_mode:
            info["tmp"] = float(self.tmp.exp().data.detach().cpu().numpy())
        self.optim_policy.step()
        if self.auto_tmp_mode:
            self.optim_tmp.step()
        self.zero_grad()
        # Logging
        info["loss/policy"] = float(policy_loss.cpu().detach().numpy())
        info["loss/policy_reg"] = float(policy_reg.cpu().detach().numpy())
        info["loss/policy_sto_reg"] = float(
            policy_stochastic_reg.cpu().detach().numpy()
        )
        if self.auto_tmp_mode:
            info["loss/tmp"] = float(tmp_loss.cpu().detach().numpy())

        info["entropy"] = float(entropy.cpu().detach().numpy())
        if self.use_lap:
            assert isinstance(replay_buffer, LAPReplayBuffer)
            replay_buffer.update_priority(priority)
        # Update target network.
        self.update_target_value_fn()
        self.n_runs += 1
        return info


def run_sac(
    rl_run_name: str,
    env_id: str,
    seed: int = 777,
    auto_tmp: bool = False,
    tmp: float = 0.2,
    action_fn: str = "ReLU",
    reward_scale: float = 1.0,
    discount_factor: float = 0.99,
    use_checkpoint: bool = False,
    use_lap: bool = False,
    replay_buffer_size: int = 1_000_000,
    **kwargs,
) -> None:
    """Run Heating Environment."""
    params = convert_dict_as_param(deepcopy(locals()))
    params["rl_alg"] = "SAC"
    print("-" * 5 + "[SAC]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    rl_run_name = f"{rl_run_name}-{timestamp}"
    base_dir = SAVE_DIR / "SAC" / rl_run_name
    base_dir.mkdir(exist_ok=True, parents=True)

    # TODO: Replace logger by mlflow.
    logger = setup_logger(str(base_dir / "training.log"))

    # Write out configuration file.
    with open(base_dir / "config.json", "w") as file_handler:
        json.dump(params, file_handler, indent=4)

    # Make envs
    env = gym.make(env_id)
    # env = RecordEpisodeStatistics(env, deque_size=1)
    replay_buffer = (
        LAPReplayBuffer(replay_buffer_size, env.observation_space, env.action_space)
        if use_lap
        else SimpleReplayBuffer(replay_buffer_size)
    )
    tmp = "auto" if auto_tmp else tmp
    agent = SAC(
        env.action_space,
        env.observation_space,
        action_fn=action_fn,
        reward_scale=reward_scale,
        discount_factor=discount_factor,
        tmp=tmp,
        **kwargs,
    )
    if use_checkpoint:
        run_rl_w_checkpoint(env, agent, logger, base_dir, replay_buffer, **kwargs)
    else:
        run_rl(env, agent, logger, base_dir, replay_buffer, seed=seed, **kwargs)
