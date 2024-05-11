"""TD7 with World Model."""


from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from rl.agent.abc import Agent
from rl.nn.abc import ACTOR, CRITIC, WORLDMODEL
from rl.nn.world_model import LatentActor, LatentCritic, WorldModel
from rl.replay_memory.lap import LAPReplayMemory
from rl.replay_memory.simple import SimpleReplayMemory
from rl.runner.run_w_checkpoint import run_rl_w_ckpt
from rl.sampler import Sampler
from rl.utils import make_env
from rl.utils.annotation import ACTION, BATCH, DONE, REWARD, STATE
from rl.utils.miscellaneous import (
    convert_dict_as_param,
    fix_seed,
    get_action_bias_scale,
    get_state_action_dims,
)
from rl.vars import SAVE_DIR


class MPC(Agent, Sampler):
    """World Model inspired by TD7."""

    def __init__(
        self,
        env_id: str,
        discount_factor: float = 0.99,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        world_model_lr: float = 3e-4,
        target_update_rate: int = 250,
        exploration_noise: float = 0.1,
        plan_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        use_lap: bool = False,
        **kwargs,
    ) -> None:
        """Init."""
        # Make Neural Networks.
        state_dim, action_dim = get_state_action_dims(env_id)
        self.policy, self.q1, self.q2, self.world_model = self.make_nn(
            state_dim, action_dim
        )
        self.target_policy = deepcopy(self.policy)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.fixed_world_model = deepcopy(self.world_model)
        self.fixed_world_model_target = deepcopy(self.world_model)

        # Save configuration
        self.action_bias, self.action_scale = get_action_bias_scale(env_id)
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.target_policy_noise = target_policy_noise
        self.exploration_noise = exploration_noise
        self.plan_noise = plan_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.use_lap = use_lap

        self.n_runs = 0
        self.device = torch.device("cpu")
        self.to(self.device)
        self.make_optimizers(policy_lr, critic_lr, world_model_lr)

        # Value CLipping
        self.value_max = -1e8
        self.value_min = 1e8
        self.value_target_max = 0
        self.value_target_min = 0

        # Plan
        self._prev_mean: torch.Tensor

    def make_nn(
        self, state_dim: int, action_dim: int
    ) -> tuple[ACTOR, CRITIC, CRITIC, WORLDMODEL]:
        """Make World Model."""
        world_model = WorldModel(state_dim, action_dim)
        policy = LatentActor(action_dim)
        q1 = LatentCritic(action_dim)
        q2 = LatentCritic(action_dim)
        return policy, q1, q2, world_model

    def make_optimizers(
        self,
        policy_lr: float,
        critic_lr: float,
        world_model_lr: float,
    ) -> None:
        """Make optimizers."""
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.optim_q_fns = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.optim_world_model = torch.optim.Adam(
            self.world_model.parameters(), lr=world_model_lr
        )

    def to(self, device: torch.device) -> "MPC":
        """Attach device."""
        self.policy = self.policy.to(device)
        self.target_policy = self.policy.to(device)
        self.q1 = self.q1.to(device)
        self.q2 = self.q2.to(device)
        self.target_q1 = self.target_q1.to(device)
        self.target_q2 = self.target_q2.to(device)

        self.world_model = self.world_model.to(device)
        self.fixed_world_model = self.fixed_world_model.to(device)
        self.fixed_world_model_target = self.fixed_world_model_target.to(device)
        self.device = device
        return self

    def load_state_dict(self, agent: "MPC") -> "MPC":
        """Load state dict."""
        self.q1.load_state_dict(agent.q1.state_dict())
        self.q2.load_state_dict(agent.q2.state_dict())
        self.policy.load_state_dict(agent.policy.state_dict())
        self.fixed_world_model.load_state_dict(agent.fixed_world_model.state_dict())
        self.world_model.load_state_dict(agent.world_model.state_dict())
        self.fixed_world_model_target.load_state_dict(
            agent.fixed_world_model_target.state_dict()
        )
        self.target_q1.load_state_dict(agent.target_q1.state_dict())
        self.target_q2.load_state_dict(agent.target_q2.state_dict())

    def zero_grad(self) -> None:
        """Apply zero gradient."""
        self.optim_q_fns.zero_grad()
        self.optim_policy.zero_grad()
        self.optim_world_model.zero_grad()

    def sample(
        self,
        state: STATE,
        deterministic: bool = False,
        use_plan: bool = False,
        is_first_obs: bool = False,
        **kwargs,
    ) -> ACTION:
        """Sample action."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.Tensor(state).float()
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            if deterministic:
                # if use_plan:
                action_traj = self._plan(state, is_first_obs).detach()
                action = action_traj[0]
            else:
                action = self._inference_action(state)
            if not deterministic:
                noise = torch.randn_like(action) * self.exploration_noise
                action += noise.to(self.device)
            action = action.cpu().detach().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
        action = action * self.action_scale + self.action_bias
        return action

    def _inference_action(self, state: STATE) -> ACTION:
        """Only forward with policy."""
        state_embedding = self.fixed_world_model.encode_state(state)
        action = self.policy.inference_mean(state_embedding)
        return action

    def _estimate_action_traj(
        self, action_traj: torch.Tensor, inital_latent: torch.Tensor
    ) -> torch.Tensor:
        """Estimate action trajectories.

        Args:
            - action_traj: (Seq, Batch, ADim)
            - initial_latent: (Batch, ADim)

        Return:
            - value: (Batch, 1)
        """
        plan_horizon = action_traj.shape[0]
        latent = inital_latent
        rewards = 0.0
        for idx in range(plan_horizon - 1):
            reward = self.world_model.predict_reward(latent, action_traj[idx])
            latent = self.fixed_world_model.encode_state_action(
                latent, action_traj[idx]
            )
            rewards = rewards + self.discount_factor**idx * reward

        # action = self.policy.inference_mean(latent)
        action = action_traj[-1]
        next_latent = self.fixed_world_model.encode_state_action(latent, action)
        next_value01 = self.q1.estimate_q_value(action, latent, next_latent)
        next_value02 = self.q2.estimate_q_value(action, latent, next_latent)
        next_value = torch.min(next_value01, next_value02)
        return rewards + self.discount_factor ** (plan_horizon) * next_value

    def _plan(self, state: torch.Tensor, is_first_obs: bool = False) -> torch.Tensor:
        """Plan.

        Arg:
            - state: (1, OBSDIM)

        Return:
            - trajectory: (Horizon, ActionDIM)

        """
        # Sample action trajectories from prior policy.
        ## initial state evolves with action traj: a1 -> a2, ... -> aN.
        trajs = 25
        n_prior_trajs = 24
        random_trajs = trajs - n_prior_trajs
        plan_horizon = 2
        mppi_steps = 1
        n_elites = 12
        temperature = 0.5
        plan_noise = self.plan_noise
        action_dim = self.action_scale.shape[0]

        ## (1, OBSDIM) -> (N, OBSDIM)
        initial_latent = self.fixed_world_model.encode_state(state)
        latent = initial_latent.repeat(n_prior_trajs, 1)

        action_traj = torch.empty(plan_horizon, trajs, action_dim, device=self.device)
        # [ 0.8156, -0.9896,  0.0294, -0.9487, -0.8471,  0.9830,  0.9524,  0.3463]
        for idx in range(plan_horizon):
            # TODO: Conflict with Original TD7.
            _action = self.policy.inference_mean(latent)
            noise = (torch.randn_like(_action) * plan_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action = (_action + noise).clamp(-1.0, 1.0)

            action_traj[idx, :n_prior_trajs] = action
            latent = self.fixed_world_model.encode_state_action(latent, action)

        # Iterate MPPI
        mean = torch.zeros(plan_horizon, action_dim).to(self.device)
        if not is_first_obs and hasattr(self, "_prev_mean"):
            mean[:-1] = self._prev_mean[1:]

        for idx in range(mppi_steps):
            # Sample action.
            ## [Input]: mean: (Horizon, AD)
            ## [Input]: std: (Horizon, AD)
            ## [Output]: random_actions: (Horizon, # of trajs - # of prior trajs, AD)
            repeated_mean = torch.stack([mean] * random_trajs, 1)
            noise = (
                torch.randn_like(repeated_mean, device=self.device) * self.plan_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            random_action = repeated_mean + noise
            random_action = random_action.clamp(-1.0, 1.0)
            action_traj[:, n_prior_trajs:] = random_action

            # Compute Elite actions.
            ## [Input]: latent: (# of saples, D)
            ## [Input]: actions: (Horizon, # of samples, AD)
            ## [Output]: value: (# of samples, 1)
            ## [Output]: elite_value: (# of elites, 1)
            ## [Output]: elite_actions: (Horizon, # of elites, AD)
            value = self._estimate_action_traj(
                action_traj, initial_latent.repeat(trajs, 1)
            )
            elite_indices = torch.topk(value.squeeze(1), n_elites, dim=0).indices
            elite_value, elite_actions = (
                value[elite_indices],
                action_traj[:, elite_indices],
            )

            # Calculate Scores
            ## [Input]: elite_value: (# of elite, 1)
            ## [Output]: score: (# of elite, 1)
            max_value = elite_value.max(0)[0]
            score = torch.exp(temperature * (elite_value - max_value))
            score = score / score.sum(0)

            # Calculate Mean and Std
            ## [Input]: score: (# of elite, 1)
            ## [Input]: elite_actions: (Horizon, # of elite, AD)
            ## [Output]: mean: (Horizon, AD)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-6
            )
        elite_action = elite_actions[:, elite_value.argmax()]
        print(elite_action)
        print("-" * 25)
        # Select Action from sample action trajectories.
        ## [Input]: score: (# of elites, 1)
        ## [Input]: elite_actions: (Horizon, # of elites, AD)
        ## [Output]: trajectory: (Horizon, AD)
        np_score = score.squeeze(1).cpu().numpy()
        trajectory = elite_actions[
            :, np.random.choice(np.arange(np_score.shape[0]), p=np_score)
        ]
        # trajectory = elite_actions[
        #     :, (-np_score).argmax()
        # ]
        self._prev_mean = mean
        return trajectory

    def _encoder_train_ops(
        self, state: STATE, action: ACTION, next_state: STATE, **kwargs
    ) -> torch.Tensor:
        """Encoder ops."""
        with torch.no_grad():
            next_state_embedding = self.world_model.encode_state(next_state)
        state_embedding = self.world_model.encode_state(state)
        state_action_embedding = self.world_model.encode_state_action(
            state_embedding, action
        )
        loss = (state_action_embedding - next_state_embedding).pow(2.0).mean()
        return loss

    def _reward_predictor_train_ops(
        self, state: STATE, action: ACTION, reward: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Reward Predictor ops."""
        with torch.no_grad():
            latent = self.fixed_world_model.encode_state(state)
        pred_reward = self.world_model.predict_reward(latent, action)
        loss = (pred_reward - reward).pow(2.0).mean()
        return loss

    @staticmethod
    def _lap_huber(td_error: torch.Tensor, min_priority: float = 1.0) -> torch.Tensor:
        """ "Caluclate Huber loss for LAP."""
        return (
            torch.where(
                td_error < min_priority, 0.5 * td_error.pow(2), min_priority * td_error
            )
            .sum(1)
            .mean()
        )

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
            next_state_embedding = self.fixed_world_model_target.encode_state(
                next_state
            )

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (
                self.target_policy.inference_mean(next_state_embedding) + noise
            ).clamp(-1.0, 1.0)
            next_state_action_embedding = (
                self.fixed_world_model_target.encode_state_action(
                    next_state_embedding, next_action
                )
            )

            next_q1 = self.target_q1.estimate_q_value(
                next_action,
                next_state_embedding,
                next_state_action_embedding,
            )
            next_q2 = self.target_q2.estimate_q_value(
                next_action,
                next_state_embedding,
                next_state_action_embedding,
            )
            next_value = torch.cat([next_q1, next_q2], -1)
            next_value = next_value.min(1, keepdim=True)[0].clamp(
                self.value_target_min, self.value_target_max
            )
            q_target = reward + self.discount_factor * next_value * done

            self.value_max = max(self.value_max, q_target.max())
            self.value_min = min(self.value_min, q_target.min())

            state_embedding = self.fixed_world_model.encode_state(state)
            state_action_embedding = self.fixed_world_model.encode_state_action(
                state_embedding, action
            )
        # calculate q value
        q1 = self.q1.estimate_q_value(action, state_embedding, state_action_embedding)
        q2 = self.q2.estimate_q_value(action, state_embedding, state_action_embedding)
        if self.use_lap:
            td_loss1 = (q1 - q_target).abs()
            td_loss2 = (q2 - q_target).abs()
            # Batch, 2
            td_loss = torch.cat([td_loss1, td_loss2], 1)
            q_loss = self._lap_huber(td_loss)
            # TODO: It is hard-coding
            priority = td_loss.max(1)[0].clamp(1.0).pow(0.4).view(-1)
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
        state_embedding = self.fixed_world_model.encode_state(state)
        state_action_embedding = self.fixed_world_model.encode_state_action(
            state_embedding, action
        )

        q1 = self.q1.estimate_q_value(action, state_embedding, state_action_embedding)
        q2 = self.q2.estimate_q_value(action, state_embedding, state_action_embedding)
        q_value = torch.cat([q1, q2], -1)
        policy_loss = -q_value.mean()
        return policy_loss

    @torch.no_grad()
    def hard_update_target_fns(self) -> None:
        """Update target network."""
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.fixed_world_model_target.load_state_dict(
            self.fixed_world_model.state_dict()
        )
        self.fixed_world_model.load_state_dict(self.world_model.state_dict())

    def train_ops(
        self,
        batch: BATCH,
        replay_buffer: LAPReplayMemory | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Run train ops."""
        self.n_runs += 1
        info = {}

        batch = {key: value.to(self.device) for key, value in batch.items()}
        # Update encoder.
        encoder_loss = self._encoder_train_ops(**batch)
        reward_predictor_loss = self._reward_predictor_train_ops(**batch)

        world_model_loss = encoder_loss + reward_predictor_loss
        self.optim_world_model.zero_grad()
        world_model_loss.backward()
        self.optim_world_model.step()
        info["loss/encoder"] = float(encoder_loss.item())
        info["loss/reward_predictor"] = float(reward_predictor_loss.item())
        info["loss/world_model"] = float(world_model_loss.item())

        # Update Q function.
        q_value_loss = self._q_train_ops(**batch)
        if isinstance(q_value_loss, tuple):
            q_value_loss, priority = q_value_loss
            assert isinstance(replay_buffer, LAPReplayMemory)
            replay_buffer.update_priority(priority)
        self.optim_q_fns.zero_grad()
        q_value_loss.backward()
        self.optim_q_fns.step()
        info["loss/q_fn"] = float(q_value_loss.item())

        # Update policy
        info["loss/policy"] = None
        if self.n_runs % self.policy_freq == 0:
            policy_loss = self._policy_train_ops(**batch)
            self.optim_policy.zero_grad()
            policy_loss.backward()
            self.optim_policy.step()
            info["loss/policy"] = float(policy_loss.item())

        if self.n_runs % self.target_update_rate == 0:
            # Update Target
            self.hard_update_target_fns()
            self.value_target_max = self.value_max
            self.value_target_min = self.value_min
            if self.use_lap:
                replay_buffer.reset_max_priority()
        return info


def run_td7_mpc(
    run_name: str,
    env_id: str,
    seed: int = 777,
    without_policy_checkpoint: bool = False,
    without_lap: bool = False,
    replay_buffer_size: int = 1_000_000,
    record_video: bool = False,
    show_progressbar: bool = True,
    **kwargs,
) -> None:
    """Run TD7-MPC."""
    params = convert_dict_as_param(deepcopy(locals()))
    params["rl_alg"] = "TD7-MPC"
    print("-" * 5 + "[TD7-MPC]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    run_name = f"{run_name}-{timestamp}"
    base_dir = SAVE_DIR / "TD7-MPC" / run_name

    # Make directory for saving and logging.
    base_dir.mkdir(exist_ok=True, parents=True)

    # Write out configuration file.
    with open(base_dir / "config.yaml", "w") as file_handler:
        yaml.dump(params, file_handler)

    # Set Seed.
    fix_seed(seed)
    # Make envs
    env = make_env(env_id)
    env.reset(seed=seed)

    replay_class = SimpleReplayMemory if without_lap else LAPReplayMemory
    replay_buffer = replay_class(replay_buffer_size, env_id)

    # mpc = TD7MPC(
    #     env_id,
    #     use_lap = not without_lap,
    #     **kwargs,
    # )
    mpc = MPC.load("save/TD7-MPC/Ant-v4-v0/ckpt.pkl")

    run_rl_w_ckpt(
        env,
        mpc,
        replay_buffer,
        base_dir,
        record_video=False,
        n_episodes=2,
        use_gpu=True,
    )
