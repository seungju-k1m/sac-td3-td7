"""TD7."""

import random
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

import yaml
import torch
import gymnasium as gym

from rl import SAVE_DIR
from rl.agent.abc import Agent
from rl.replay_buffer.lap import LAPReplayBuffer
from rl.replay_buffer.simple import SimpleReplayBuffer
from rl.sampler import Sampler
from rl.neural_network import Encoder, SALEActor, SALECritic
from rl.utils.annotation import ACTION, BATCH, DONE, STATE, REWARD
from rl.utils.miscellaneous import (
    convert_dict_as_param,
    get_state_action_dims,
    setup_logger,
)
from rl.runner import run_rl, run_rl_w_ckpt


class TD7(Agent, Sampler):
    """Agent for TD7."""

    def __init__(
        self,
        env_id: str,
        discount_factor: float = 0.99,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        target_update_rate: int = 250,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        use_lap: bool = False,
        **kwargs,
    ) -> None:
        """Initialize."""
        assert env_id in gym.registry
        # Make neural network.
        self.encoder, self.policy, self.q1, self.q2 = self.make_nn(env_id)
        self.target_policy = deepcopy(self.policy)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.fixed_encoder = deepcopy(self.encoder)
        self.fixed_encoder_target = deepcopy(self.encoder)

        # Save cconiguration.
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.target_policy_noise = target_policy_noise
        self.exploration_noise = exploration_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.use_lap = use_lap
        self.n_runs = 0
        self.device = torch.device("cpu")
        self.to(self.device)
        # Make optimizer.
        self.make_optimizers(policy_lr, critic_lr)

        # Value Clipping
        # Value clipping tracked values
        self.value_max = -1e8
        self.value_min = 1e8
        self.value_target_max = 0
        self.value_target_min = 0

    @staticmethod
    def make_nn(
        env_id: str, **kwargs
    ) -> tuple[Encoder, SALEActor, SALECritic, SALECritic, SALECritic]:
        """Make neurla networks."""
        state_dim, action_dim = get_state_action_dims(env_id)
        encoder = Encoder(state_dim, action_dim)
        policy = SALEActor(state_dim, action_dim)
        q1 = SALECritic(state_dim, action_dim)
        q2 = SALECritic(state_dim, action_dim)
        return encoder, policy, q1, q2

    def to(self, device: torch.device) -> None:
        """Attach device."""
        self.policy = self.policy.to(device)
        self.target_policy = self.policy.to(device)
        self.q1 = self.q1.to(device)
        self.q2 = self.q2.to(device)
        self.target_q1 = self.target_q1.to(device)
        self.target_q2 = self.target_q2.to(device)
        self.encoder = self.encoder.to(device)
        self.fixed_encoder = self.fixed_encoder.to(device)
        self.fixed_encoder_target = self.fixed_encoder_target.to(device)
        self.device = device

    def load_state_dict(self, agent: "TD7") -> None:
        """Load state dict."""
        self.q1.load_state_dict(agent.q1.state_dict())
        self.q2.load_state_dict(agent.q2.state_dict())
        self.policy.load_state_dict(agent.policy.state_dict())
        self.encoder.load_state_dict(agent.encoder.state_dict())
        self.fixed_encoder.load_state_dict(agent.fixed_encoder.state_dict())
        self.fixed_encoder_target.load_state_dict(
            agent.fixed_encoder_target.state_dict()
        )
        self.target_q1.load_state_dict(agent.target_q1.state_dict())
        self.target_q2.load_state_dict(agent.target_q2.state_dict())

    def make_optimizers(self, policy_lr: float, critic_lr: float) -> None:
        """Make optimizers."""
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=policy_lr)

    def zero_grad(self) -> None:
        """Apply zero gradient."""
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        self.optim_encoder.zero_grad()

    def sample(self, state: STATE, deterministic: bool = False, **kwargs) -> ACTION:
        """Sample action."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.Tensor(state).float()
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            action = self._inference_action(state)
            if not deterministic:
                noise = torch.randn_like(action) * self.exploration_noise
                action += noise.to(self.device)
            action = action.cpu().detach().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
        return action

    def _inference_action(self, state: STATE) -> ACTION:
        """Only forward with policy."""
        state_embedding = self.fixed_encoder.encode_state(state)
        action = self.policy.forward(state, state_embedding)
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
        """Q fn train ops."""
        # make q target
        with torch.no_grad():
            next_state_embedding = self.fixed_encoder_target.encode_state(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (
                self.target_policy.forward(next_state, next_state_embedding) + noise
            ).clamp(-1.0, 1.0)
            next_state_action_embedding = self.fixed_encoder_target.encode_state_action(
                next_state_embedding, next_action
            )

            next_q1 = self.target_q1.forward(
                next_state,
                next_action,
                next_state_action_embedding,
                next_state_embedding,
            )
            next_q2 = self.target_q2.forward(
                next_state,
                next_action,
                next_state_action_embedding,
                next_state_embedding,
            )

            next_value = torch.min(next_q1, next_q2).clamp(
                self.value_target_min, self.value_target_max
            )
            q_target = reward + self.discount_factor * next_value * done

            self.value_max = max(self.value_max, q_target.max())
            self.value_min = min(self.value_min, q_target.min())

            state_embedding = self.fixed_encoder.encode_state(state)
            state_action_embedding = self.fixed_encoder.encode_state_action(
                state_embedding, action
            )
        # calculate q value
        q1 = self.q1.forward(state, action, state_action_embedding, state_embedding)
        q2 = self.q2.forward(state, action, state_action_embedding, state_embedding)
        if self.use_lap:
            td_loss1 = (q1 - q_target).abs()
            td_loss2 = (q2 - q_target).abs()
            q_loss = self._lap_huber(td_loss1 + td_loss2)
            # TODO: It is hard-coding
            priority = torch.max(td_loss1, td_loss2).clamp(1.0).pow(0.4).view(-1)
            return q_loss, priority.cpu().detach()
        else:
            q1_loss = torch.mean((q_target - q1) ** 2.0) * 0.5
            q2_loss = torch.mean((q_target - q2) ** 2.0) * 0.5
            q_loss = q1_loss + q2_loss
            return q_loss

    def _encoder_train_ops(
        self, state: STATE, action: ACTION, next_state: STATE, **kwargs
    ) -> torch.Tensor:
        """Encoder ops."""
        with torch.no_grad():
            next_state_embedding = self.encoder.encode_state(next_state)
        state_embedding = self.encoder.encode_state(state)
        state_action_embedding = self.encoder.encode_state_action(
            state_embedding, action
        )
        loss = ((state_action_embedding - next_state_embedding).pow(2.0)).mean()
        return loss

    def _policy_train_ops(self, state: STATE, **kwargs) -> torch.Tensor:
        """Policy ops."""
        # Calculate policy loss.
        action = self._inference_action(state)
        state_embedding = self.fixed_encoder.encode_state(state)
        state_action_embedding = self.fixed_encoder.encode_state_action(
            state_embedding, action
        )

        q1 = self.q1.forward(state, action, state_action_embedding, state_embedding)
        q2 = self.q2.forward(state, action, state_action_embedding, state_embedding)
        policy_loss = -(q1.mean() + q2.mean())
        return policy_loss

    @torch.no_grad()
    def hard_update_target_fns(self) -> None:
        """Update target network."""
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
        self.fixed_encoder.load_state_dict(self.encoder.state_dict())

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
        # Update encoder.
        encoder_loss = self._encoder_train_ops(**batch)

        self.optim_encoder.zero_grad()
        encoder_loss.backward()
        self.optim_encoder.step()
        info["train/encoder"] = float(encoder_loss.cpu().detach().numpy())
        self.zero_grad()

        # Update state value function.
        # Update Action-State value function.
        q_value_loss = self._q_train_ops(**batch)
        if isinstance(q_value_loss, tuple):
            q_value_loss, priority = q_value_loss
            assert isinstance(replay_buffer, LAPReplayBuffer)
            replay_buffer.update_priority(priority)
        q_value_loss.backward()
        self.optim_q_fns.step()
        self.zero_grad()
        info["train/q_fn"] = float(q_value_loss.cpu().detach().numpy())

        # Update policy.
        info["train/policy"] = None
        if self.n_runs % self.policy_freq == 0:
            policy_loss = self._policy_train_ops(**batch)
            policy_loss.backward()
            info["train/policy"] = float(policy_loss.cpu().detach().numpy())

            self.optim_policy.step()
            self.zero_grad()
        if self.n_runs % self.target_update_rate == 0:
            # Update Target
            self.hard_update_target_fns()
            self.value_target_max = self.value_max
            self.value_target_min = self.value_min
            if self.use_lap:
                replay_buffer.reset_max_priority()
        self.n_runs += 1
        return info


def run_td7(
    rl_run_name: str,
    env_id: str,
    seed: int = 777,
    without_policy_checkpoint: bool = False,
    without_lap: bool = False,
    replay_buffer_size: int = 1_000_000,
    benchmark_idx: int = 0,
    record_video: bool = False,
    show_progressbar: bool = True,
    **kwargs,
) -> None:
    """Run Heating Environment."""
    params = convert_dict_as_param(deepcopy(locals()))
    params["rl_alg"] = "TD7"
    print("-" * 5 + "[TD7]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    if benchmark_idx > 0:
        base_dir = (
            SAVE_DIR
            / "VALID"
            / env_id
            / "TD7"
            / f"TD7-{rl_run_name}"
            / str(benchmark_idx)
        )
        show_progressbar = False
        record_video = False
    else:
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
        rl_run_name = f"{rl_run_name}-{timestamp}"
        base_dir = SAVE_DIR / "TD7" / rl_run_name

    # Make directory for saving and logging.
    base_dir.mkdir(exist_ok=True, parents=True)

    # TODO: Replace logger by mlflow.
    logger = setup_logger(str(base_dir / "training.log"))

    # Write out configuration file.
    with open(base_dir / "config.yaml", "w") as file_handler:
        yaml.dump(params, file_handler)

    # Set Seed.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Make envs
    env = gym.make(env_id)
    env.reset(seed=seed)

    replay_buffer = (
        SimpleReplayBuffer(replay_buffer_size)
        if without_lap
        else LAPReplayBuffer(
            replay_buffer_size, env.observation_space, env.action_space
        )
    )
    agent = TD7(
        env_id,
        use_lap=not without_lap,
        **kwargs,
    )
    if without_policy_checkpoint:
        run_rl(
            env,
            agent,
            replay_buffer,
            logger,
            show_progressbar=show_progressbar,
            record_video=record_video,
            seed=seed,
            **kwargs,
        )
    else:
        run_rl_w_ckpt(
            env,
            agent,
            replay_buffer,
            logger,
            seed=seed,
            record_video=record_video,
            show_progressbar=show_progressbar,
            **kwargs,
        )
