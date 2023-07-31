from typing import Literal

import torch
import gymnasium as gym
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
        env: str | gym.Env,
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
    ) -> None:
        self.env = env if isinstance(env, gym.Env) else gym.make(env)
        self.discount_factor = discount_factor
        self.hidden_sizes = hidden_sizes
        self.action_fn = action_fn
        self.auto_tmp_mode = True if tmp == "auto" else False
        self.tmp = (
            nn.Parameter(torch.ones(1).float(), requires_grad=True)
            if self.auto_tmp_mode
            else tmp
        )
        self.make_nn()
        self.make_optimizers(policy_lr, critic_lr)
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.step_per_batch = step_per_batch
        self.tau = tau
        self.action_dim = self.env.action_space.shape[0]
        self.policy_reg_coeff = policy_reg_coeff
        self.policy_prior = torch.distributions.Normal(0.0, 1.0)

    def make_nn(self) -> None:
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.policy = make_feedforward(
            obs_dim, action_dim * 2, self.hidden_sizes, self.action_fn, True
        )

        self.q1 = make_feedforward(
            obs_dim + action_dim, 1, self.hidden_sizes, self.action_fn, True
        )
        self.q2 = make_feedforward(
            obs_dim + action_dim, 1, self.hidden_sizes, self.action_fn, True
        )
        self.value_fn = make_feedforward(
            obs_dim, 1, self.hidden_sizes, self.action_fn, True
        )
        self.target_value_fn = make_feedforward(
            obs_dim, 1, self.hidden_sizes, self.action_fn, True
        )

    def make_optimizers(self, policy_lr: float, critic_lr: float) -> None:
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.optim_value = torch.optim.Adam(self.value_fn.parameters(), lr=policy_lr)
        if self.auto_tmp_mode:
            self.optim_tmp = torch.optim.Adam([self.tmp], lr=policy_lr)

    def zero_grad(self) -> None:
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        self.optim_value.zero_grad()
        if self.auto_tmp_mode:
            self.optim_tmp.zero_grad()

    def step(self) -> None:
        self.optim_q_fns.step()
        self.optim_policy.step()
        self.optim_value.step()
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
            next_value = self.target_value_fn(next_obs)
            q_target = reward / self.tmp + self.discount_factor * next_value * done
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
        action, log_pi, (mean, log_std) = self.policy_forward(obs)
        obs_action = torch.cat((obs, action), -1)
        # with torch.no_grad():
        q1, q2 = self.q1.forward(obs_action), self.q2.forward(obs_action)
        q_value = torch.min(q1, q2)
        # q_value = self.q1.forward(obs_action)
        policy_loss = torch.mean(-q_value + log_pi)
        tmp_loss = 0.0
        if self.auto_tmp_mode:
            tmp_loss = torch.mean(self.tmp.exp() * (-log_pi.detach() + self.action_dim))
        # Value at given observation with current and target.
        policy_reg = 0.5 * (torch.mean(log_std**2) + torch.mean(mean**2))
        return policy_loss, tmp_loss, policy_reg

    def _value_ops(self, obs: OBSERVATION, **kwargs) -> torch.Tensor:
        value = self.value_fn(obs)
        action, log_pi = self.policy_forward(obs)[:2]
        obs_action = torch.cat((obs, action), -1)
        q1, q2 = self.q1(obs_action), self.q2(obs_action)
        qvalue = torch.min(q1, q2)
        target = qvalue - (
            log_pi - self.policy_prior.log_prob(action).sum(-1, keepdim=True)
        )
        value_loss = torch.mean((value - target.detach()).pow(2.0))
        return value_loss

    def sample(self, obs: OBSERVATION, deterministic: bool = False, **kwargs) -> ACTION:
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
        for v_parm, target_v_param in zip(
            self.value_fn.parameters(), self.target_value_fn.parameters()
        ):
            target_v_param.copy_(self.tau * v_parm + target_v_param * (1 - self.tau))

    def train_ops(self, batch: BATCH, *args, **kwargs) -> None:
        # Update state value function.
        info = {}
        max_norm = float("inf")
        # max_norm = 1
        value_loss = self._value_ops(**batch)
        value_loss.backward()
        info["norm/value"] = calculate_grad_norm(self.value_fn)
        clip_grad_norm(self.value_fn, max_norm)
        self.optim_value.step()
        self.zero_grad()
        info["loss/value"] = float(value_loss.cpu().detach().numpy())

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
        policy_loss, tmp_loss, policy_reg = self._policy_ops(**batch)
        # loss = policy_loss + policy_reg
        loss = policy_loss + self.policy_reg_coeff * policy_reg
        if self.auto_tmp_mode:
            loss += tmp_loss
        loss.backward()
        info["norm/policy"] = calculate_grad_norm(self.policy)
        clip_grad_norm(self.policy, max_norm)
        if self.auto_tmp_mode:
            info["tmp"] = float(self.tmp.data.detach().cpu().numpy())
        self.optim_policy.step()
        if self.auto_tmp_mode:
            self.optim_tmp.step()
        self.zero_grad()
        info["loss/policy"] = float(policy_loss.cpu().detach().numpy())
        info["loss/policy_reg"] = float(policy_reg.cpu().detach().numpy())
        if self.auto_tmp_mode:
            info["loss/tmp"] = float(tmp_loss.cpu().detach().numpy())

        # Update target network.
        self.update_target_value_fn()

        return info
