from copy import deepcopy
from typing import Literal

import torch
from torch import nn

from rl.core.agent import Agent
from rl.core.sampler import Sampler
from rl.utils.neural_network import (
    calculate_grad_norm,
    clip_grad_norm,
    make_feedforward,
)
from rl.utils.annotation import ACTION, BATCH, DONE, EPS, OBSERVATION, REWARD


class SAC(Agent, Sampler):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
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
        policy_reg_coeff: float = 1e-3,
        device: str = "mps",
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.hidden_sizes = hidden_sizes
        self.action_fn = action_fn
        self.auto_tmp_mode = True if tmp == "auto" else False
        self.tmp = (
            nn.Parameter(torch.zeros(1).float().to(self.device), requires_grad=True)
            if self.auto_tmp_mode
            else tmp
        )
        self.make_nn()
        self.make_optimizers(policy_lr, critic_lr)
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.step_per_batch = step_per_batch
        self.tau = tau
        self.policy_reg_coeff = policy_reg_coeff
        self.policy_prior = torch.distributions.Normal(0.0, 1.0)

    def make_nn(self) -> None:
        obs_dim, action_dim = self.obs_dim, self.action_dim
        self.policy = make_feedforward(
            obs_dim,
            action_dim * 2,
            self.hidden_sizes,
            self.action_fn,
            True,
            self.device,
        )

        self.q1 = make_feedforward(
            obs_dim + action_dim,
            1,
            self.hidden_sizes,
            self.action_fn,
            True,
            self.device,
        )
        self.q2 = make_feedforward(
            obs_dim + action_dim,
            1,
            self.hidden_sizes,
            self.action_fn,
            True,
            self.device,
        )
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def make_optimizers(self, policy_lr: float, critic_lr: float) -> None:
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        if self.auto_tmp_mode:
            self.optim_tmp = torch.optim.Adam([self.tmp], lr=policy_lr)

    def zero_grad(self) -> None:
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        if self.auto_tmp_mode:
            self.optim_tmp.zero_grad()

    def step(self) -> None:
        self.optim_q_fns.step()
        self.optim_policy.step()
        if self.auto_tmp_mode:
            self.optim_tmp.step()

    def policy_forward(
        self, obs: OBSERVATION
    ) -> tuple[ACTION, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    ) -> torch.Tensor:
        # make q target
        with torch.no_grad():
            # next_value = self.target_value_fn.forward(next_obs)
            next_action, next_log_pi = self.policy_forward(next_obs)[:2]
            next_obs_action = torch.cat((next_obs, next_action), -1)
            next_value = torch.min(
                self.target_q1.forward(next_obs_action),
                self.target_q2.forward(next_obs_action),
            )
            tmp = self.tmp.exp() if self.auto_tmp_mode else self.tmp
            q_target = (
                reward + self.discount_factor * (next_value - tmp * next_log_pi) * done
            )
        # calculate q value
        obs_action = torch.cat((obs, action), 1)
        q1, q2 = self.q1.forward(obs_action), self.q2.forward(obs_action)
        q1_loss = torch.mean((q_target - q1) ** 2.0) * 0.5
        q2_loss = torch.mean((q_target - q2) ** 2.0) * 0.5
        q_loss = q1_loss + q2_loss
        return q_loss

    def _policy_ops(
        self, obs: OBSERVATION, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """Policy ops."""
        # Calculate policy loss.
        action, log_pi, (mean, log_std) = self.policy_forward(obs)
        obs_action = torch.cat((obs, action), -1)
        q1, q2 = self.q1.forward(obs_action), self.q2.forward(obs_action)
        q_value = torch.min(q1, q2)
        tmp = self.tmp.exp().detach() if self.auto_tmp_mode else self.tmp
        policy_loss = torch.mean(-q_value + log_pi * tmp)

        # Calcuate tmp loss.
        tmp_loss = (
            torch.mean(self.tmp.exp() * (-log_pi.detach() + self.action_dim))
            if self.auto_tmp_mode
            else 0.0
        )

        # Entropy for logging.
        entropy = -(log_pi.mean().detach())

        # Regularizer for policy and logging.
        policy_reg = 0.5 * (torch.mean(log_std**2) + torch.mean(mean**2))
        return policy_loss, tmp_loss, policy_reg, entropy

    def sample(self, obs: OBSERVATION, deterministic: bool = False, **kwargs) -> ACTION:
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.Tensor(obs).float().to(self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            mean, log_std = self.policy_forward(obs)[-1]
            if deterministic:
                action = torch.tanh(mean)
            else:
                action = torch.tanh(
                    torch.distributions.Normal(mean, log_std.exp()).sample()
                )
            action = action.cpu().detach().numpy()
            if action.shape[0] == 1:
                action = action[0]
        return action

    @torch.no_grad()
    def update_target_value_fn(self) -> None:
        """Update target value function."""
        for q_parm, t_q_parm in zip(self.q1.parameters(), self.target_q1.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))
        for q_parm, t_q_parm in zip(self.q2.parameters(), self.target_q2.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))

    def train_ops(self, batch: BATCH, *args, **kwargs) -> None:
        # Update state value function.
        info = {}
        max_norm = float("inf")
        # Update Action-State value function.
        q_value_loss = self._q_value_ops(**batch)
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
        loss = policy_loss + self.policy_reg_coeff * policy_reg
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
        if self.auto_tmp_mode:
            info["loss/tmp"] = float(tmp_loss.cpu().detach().numpy())

        info["entropy"] = float(entropy.cpu().detach().numpy())
        # Update target network.
        self.update_target_value_fn()

        return info
