"""Soft Actor Critic."""

from typing import Callable
import pandas as pd
from copy import deepcopy
from datetime import datetime

import yaml
import torch
import gymnasium as gym
from torch import nn

from rl import SAVE_DIR
from rl.agent import Agent
from rl.nn.abc import ACTOR, CRITIC
from rl.replay_memory import SimpleReplayMemory, LAPReplayMemory, REPLAYMEMORY
from rl.sampler import Sampler
from rl.runner import run_rl
from rl.utils import convert_dict_as_param, get_state_action_dims
from rl.utils.annotation import ACTION, BATCH, DONE, EPS, STATE, REWARD
from rl.utils.miscellaneous import fix_seed, get_action_bias_scale
from rl.nn import MLPActor, MLPCritic, annotate_make_nn


class SAC(Agent, Sampler):
    """Agent for Soft Actor Critic."""

    def __init__(
        self,
        env_id: str,
        discount_factor: float = 0.99,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        tau: float = 0.005,
        tmp: float = -1.0,
        use_lap: bool = False,
        max_grad_norm: float = float("inf"),
        make_nn: Callable | None = None,
        **make_nn_kwargs,
    ) -> None:
        """Initialize."""
        # Make policy and q functions.
        state_dim, action_dim = get_state_action_dims(env_id)
        if make_nn is None:
            self.policy, self.q1, self.q2 = self.make_nn(state_dim, action_dim)
        else:
            make_nn_kwargs["state_dim"] = state_dim
            make_nn_kwargs["action_dim"] = action_dim
            self.policy, self.q1, self.q2 = annotate_make_nn(make_nn)(**make_nn_kwargs)
        # Copy target q fns.
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)

        # Set trainable temperature variable.
        self.auto_tmp_mode = True if tmp < 0.0 else False
        self.tmp = (
            nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            if tmp < 0.0
            else tmp
        )

        # Prepare optimizers corresponding to policy, qfns and temperature.
        self.optim_policy, self.optim_q_fns, self.optim_tmp = self.make_optimizers(
            policy_lr, critic_lr
        )
        if self.auto_tmp_mode:
            self.target_entropy = -action_dim

        # Store hyper-parameter used during training.
        self.device = torch.device("cpu")
        self.discount_factor = discount_factor
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.tau = tau
        self.n_runs = 0
        self.use_lap = use_lap
        self.max_grad_norm = max_grad_norm
        self.action_bias, self.action_scale = get_action_bias_scale(env_id)

    @staticmethod
    def make_nn(state_dim: int, action_dim: int) -> tuple[ACTOR, CRITIC, CRITIC]:
        """Make neural networks."""
        policy = MLPActor(state_dim, action_dim * 2)
        critic01 = MLPCritic(state_dim, action_dim)
        critic02 = MLPCritic(state_dim, action_dim)
        return policy, critic01, critic02

    def to(self, device: torch.device) -> None:
        """Attatch device."""
        self.q1 = self.q1.to(device)
        self.q2 = self.q2.to(device)
        self.policy = self.policy.to(device)

        self.target_q1 = self.target_q1.to(device)
        self.target_q2 = self.target_q2.to(device)

        if self.auto_tmp_mode:
            self.tmp = self.tmp.to(device)
        self.device = device

    def load_state_dict(self, agent: "SAC") -> None:
        """Load state dict."""
        self.q1.load_state_dict(agent.q1.state_dict())
        self.q2.load_state_dict(agent.q2.state_dict())
        self.policy.load_state_dict(agent.policy.state_dict())
        self.target_q1.load_state_dict(agent.target_q1.state_dict())
        self.target_q2.load_state_dict(agent.target_q2.state_dict())

    def make_optimizers(
        self, policy_lr: float, critic_lr: float
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer | None
    ]:
        """Make optimizers."""
        optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        if self.auto_tmp_mode:
            optim_tmp = torch.optim.Adam([self.tmp], lr=policy_lr)
        else:
            optim_tmp = None
        return optim_policy, optim_q_fns, optim_tmp

    def zero_grad(self) -> None:
        """Apply zero gradient."""
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        if self.auto_tmp_mode:
            self.optim_tmp.zero_grad()

    @torch.no_grad()
    def sample(self, state: STATE, deterministic: bool = False, **kwargs) -> ACTION:
        """Sample action."""
        # (Optional) Convert state into torch.Tensor
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        # Generate action via policy.
        distribution = self._inference(state)
        if deterministic:
            action = torch.tanh(distribution.mean)
        else:
            action = self._rsample(distribution)[0]

        # Convert it as numpy for interfacing `gym.Env``.
        action = action.cpu().detach().numpy()[0]
        action = action * self.action_scale + self.action_bias
        return action

    def _inference(self, state: STATE) -> torch.distributions.Normal:
        """Inference policy distribution."""
        mean, log_std = self.policy.inference_mean_logvar(state)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        return distribution

    @staticmethod
    def _rsample(
        distribution: torch.distributions.Normal,
    ) -> tuple[ACTION, torch.Tensor]:
        """Re-parameterization tricks."""
        arctanh_action = distribution.rsample()
        action = torch.tanh(arctanh_action)

        # Calculate log-probability of sample.
        log_pi = distribution.log_prob(arctanh_action).sum(-1, keepdim=True)
        log_pi = log_pi - torch.log(1 - action.pow(2.0) + EPS).sum(-1, keepdim=True)
        return action, log_pi

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
            next_state_dist = self._inference(next_state)
            next_action, next_log_pi = self._rsample(next_state_dist)
            next_target_q1 = self.target_q1.estimate_q_value(next_state, next_action)
            next_target_q2 = self.target_q2.estimate_q_value(next_state, next_action)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            tmp = self.tmp.exp() if self.auto_tmp_mode else self.tmp
            q_target = (
                reward
                + self.discount_factor * (next_target_q - tmp * next_log_pi) * done
            )
        # calculate q value
        q1 = self.q1.estimate_q_value(state, action)
        q2 = self.q2.estimate_q_value(state, action)

        # If LAP Replayer is used, additionally return priority.
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

    def _policy_train_ops(
        self, state: STATE, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | float, torch.Tensor]:
        """Run Train ops: Policy."""
        # Inference policy distribution.
        distribution = self._inference(state)

        # R-Sample.
        action, log_pi = self._rsample(distribution)

        # Calculate policy objective.
        q1 = self.q1.estimate_q_value(state, action)
        q2 = self.q2.estimate_q_value(state, action)
        q_value = torch.min(q1, q2)
        tmp = self.tmp.exp().detach() if self.auto_tmp_mode else self.tmp
        policy_obj = torch.mean(-q_value + log_pi * tmp)

        # Calcuate tmp loss.
        tmp_obj = (
            torch.mean(self.tmp.exp() * (-log_pi.detach() - self.target_entropy))
            if self.auto_tmp_mode
            else 0.0
        )

        # Entropy for logging.
        entropy = -(log_pi.mean().detach())

        return policy_obj, tmp_obj, entropy

    @torch.no_grad()
    def _update_target_qfns(self) -> None:
        """Update target value function."""
        for q_parm, t_q_parm in zip(self.q1.parameters(), self.target_q1.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))
        for q_parm, t_q_parm in zip(self.q2.parameters(), self.target_q2.parameters()):
            t_q_parm.copy_(self.tau * q_parm + t_q_parm * (1 - self.tau))

    def train_ops(
        self, batch: BATCH, replay_buffer: REPLAYMEMORY | None = None, *args, **kwargs
    ) -> None:
        """Run train ops."""
        info = {}
        batch = {key: value.to(self.device) for key, value in batch.items()}

        # Train Q functions.
        q_value_loss = self._q_train_ops(**batch)
        if isinstance(q_value_loss, tuple):
            q_value_loss, priority = q_value_loss
            assert isinstance(replay_buffer, LAPReplayMemory)
            replay_buffer.update_priority(priority)

        q_value_loss.backward()
        self.optim_q_fns.step()
        self.zero_grad()
        info["train/q_fn"] = float(q_value_loss.cpu().detach().numpy())

        # Train policy.
        policy_obj, tmp_loss, entropy = self._policy_train_ops(**batch)
        obj = policy_obj
        if self.auto_tmp_mode:
            obj += tmp_loss
        obj.backward()

        # (Optional) Train temperatgure.
        if self.auto_tmp_mode:
            info["tmp"] = float(self.tmp.exp().data.detach().cpu().numpy())
            info["norm/tmp"] = float(self.tmp.grad.data)
        self.optim_policy.step()
        if self.auto_tmp_mode:
            self.optim_tmp.step()
        self.zero_grad()

        # info contains result for logging.
        info["train/policy"] = float(policy_obj.cpu().detach().numpy())
        if self.auto_tmp_mode:
            info["train/tmp"] = float(tmp_loss.cpu().detach().numpy())
        info["entropy"] = float(entropy.cpu().detach().numpy())

        # Update target network.
        self._update_target_qfns()
        self.n_runs += 1
        return info

    def __repr__(self) -> str:
        """Return Algorithm Name."""
        return "SAC"


def run_sac(
    run_name: str,
    env_id: str,
    use_lap: bool = False,
    replay_buffer_size: int = 1_000_000,
    record_video: bool = False,
    seed: int = 777,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Print-out Current local params.
    params = convert_dict_as_param(deepcopy(locals()))
    params["rl_alg"] = "SAC"
    print("-" * 5 + "[SAC]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()

    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    run_name = f"{run_name}-{timestamp}"
    base_dir = SAVE_DIR / "SAC" / run_name

    # Make directory for saving and logging.
    base_dir.mkdir(exist_ok=True, parents=True)

    # Write out configuration file.
    with open(base_dir / "config.yaml", "w") as file_handler:
        yaml.dump(params, file_handler)
    # Set Seed.
    fix_seed(seed)

    # Make envs.
    env = gym.make(env_id)
    env.reset(seed=seed)

    replay_class = LAPReplayMemory if use_lap else SimpleReplayMemory
    replay_buffer = replay_class(replay_buffer_size, env_id)

    agent = SAC(
        env_id,
        **kwargs,
    )
    run_rl(
        env,
        agent,
        replay_buffer,
        base_dir,
        record_video=record_video,
        **kwargs,
    )
