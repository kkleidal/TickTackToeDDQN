import copy
import datetime
import os
from contextlib import ExitStack
from typing import Dict, Optional

import numpy as np
import torch
import tqdm

import wandb
from arena import AgentArena
from models import TransformerTickTackToeModel
from replay_buffer import CircularBuffer
from ttt_env import (
    MultiPolicy,
    OptimalPolicy,
    Policy,
    RandomPolicy,
    TickTackToeEnvironment,
)
from utils import EMAMeter, maybe_to_cuda, random_of_max


class DDQNCallback:
    """
    This class represents a callback for the DDQNTrainer.

    The callback methods are called at various points during training.
    """

    def on_fit_start(self, trainer: "DDQNTrainer"):
        """
        This method is called when the training starts.

        Args:
            trainer (DDQNTrainer): The trainer instance.
        """
        ...

    def on_episode_start(self, trainer: "DDQNTrainer"):
        """
        This method is called when a new episode starts.

        Args:
            trainer (DDQNTrainer): The trainer instance.
        """
        ...

    def on_training_update(
        self,
        trainer: "DDQNTrainer",
        loss: torch.Tensor,
        predicted_q_vals: torch.Tensor,
        target_q_vals: torch.Tensor,
        initial_states: torch.Tensor,
        actions: torch.Tensosr,
        rewards: torch.Tensor,
        next_states: list[torch.Tensor],
    ):
        """
        This method is called when the model is updated.

        Args:
            trainer (DDQNTrainer): The trainer instance.
            loss (torch.Tensor): The unpooled loss tensor.
            predicted_q_vals (torch.Tensor): The predicted q values.
            target_q_vals (torch.Tensor): The target q values.
            initial_states (torch.Tensor): The initial states.
            actions (torch.Tensor): The actions.
            rewards (torch.Tensor): The rewards.
            next_states (list[torch.Tensor]): The next states.
        """
        ...

    def on_update_target(self, trainer: "DDQNTrainer"):
        """
        This method is called when the target model is updated with weights from the primary model.

        Args:
            trainer (DDQNTrainer): The trainer instance.
        """
        ...

    def on_episode_end(self, trainer: "DDQNTrainer", episode_history: list):
        """
        This method is called when an episode ends.

        Args:
            trainer (DDQNTrainer): The trainer instance.
            episode_history (list): The episode history containing tuples of events
        """
        ...

    def on_update_epsilon(self, trainer: "DDQNTrainer", epsilon):
        """
        This method is called when the epsilon value is updated.

        Args:
            trainer (DDQNTrainer): The trainer instance.
            epsilon (float): The new epsilon value.
        """
        ...

    def on_fit_end(self, trainer: "DDQNTrainer"):
        """
        This method is called when the training ends.

        Args:
            trainer (DDQNTrainer): The trainer instance.
        """
        ...


class DDQNTrainer:
    """
    This class represents a trainer for the Double DQN algorithm.

    Args:
        env: The environment to train in.
        model: The model to train.
        opt: The optimizer to use for training.
        logdir (str, optional): The directory to log training information. Defaults to None.
        replay_buffer_len (int, optional): The length of the replay buffer. Defaults to 1_000_000.
        max_training_steps (int, optional): The maximum number of training steps. Defaults to 200_000.
        update_target_every (int, optional): The frequency of target updates. Defaults to 20_000.
        gamma (float, optional): The discount factor. Defaults to 0.55.
        learning_starts (int, optional): The number of steps before learning starts. Defaults to 10_000.
        target_start (int, optional): The number of steps before target updates start. Defaults to 30_000.
        batch_size (int, optional): The size of the training batch. Defaults to 128.
        train_freq (int, optional): The frequency of training. Defaults to 4.
    """

    def __init__(
        self,
        env,
        model,
        opt,
        logdir=None,
        replay_buffer_len=1_000_000,
        max_training_steps=200_000,
        update_target_every=20_000,
        gamma=0.55,
        learning_starts=10_000,
        target_start=30_000,
        batch_size=128,
        train_freq=4,
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        gradient_clipping=None,
        callbacks: list[DDQNCallback] = [],
        logger=None,
    ):
        if logdir is None:
            logdir = f"model_logs/{datetime.datetime.now().isoformat()}"
            os.makedirs(logdir)
        self.logdir = logdir
        self.logger = logger

        self.env = env
        self.model = model
        self.opt = opt

        self.replay_buffer_len = replay_buffer_len
        self.max_training_steps = max_training_steps
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.target_start = target_start
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.exploration_update_freq = int(exploration_fraction * max_training_steps)
        self.gradient_clipping = gradient_clipping

        self.callbacks = callbacks

        self.global_step = None
        self.last_train_update = 0
        self.last_exploration_update = 0
        self.target_initialized = False
        self.last_target_update = 0
        self.target_model = None
        self.epsilon = None
        self.replay_buffer = None

    def log_dict(self, d: Dict[str, float]):
        """Log a dictionary of values to the logger."""
        if self.logger:
            self.logger.log_dict(d, self.global_step)

    def _event(self, event_name: str, *args, **kwargs):
        """Call an event on all callbacks."""
        for callback in self.callbacks:
            getattr(callback, event_name)(self, *args, **kwargs)

    def fit(self):
        """Fit the model."""

        self.global_step = 0
        self.last_train_update = 0
        self.last_exploration_update = 0
        self.target_initialized = False
        self.last_target_update = 0

        self.target_model = maybe_to_cuda(copy.deepcopy(self.model))
        self.model = maybe_to_cuda(self.model)
        self.epsilon = self.exploration_initial_eps
        self.replay_buffer = CircularBuffer(self.replay_buffer_len)

        self._event("on_fit_start")
        while self.global_step < self.max_training_steps:
            self._train_episode()
        self._event("on_fit_end")

    def _take_action(self, board, epsilon, next_featurized, episode_history):
        """Take an action in the environment during training."""
        valid_actions = self.env.get_valid_actions(board)
        featurized = next_featurized
        if np.random.rand() < epsilon:
            action = valid_actions[np.random.randint(len(valid_actions))]
            episode_history.append(("action", "random", action))
        else:
            self.model.eval()
            q_val = (
                self.model(maybe_to_cuda(featurized).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .detach()
                .numpy()
            )
            q_val_valid = q_val[valid_actions]
            # check_soft_divergence(q_val_valid, gamma)
            action = valid_actions[random_of_max(q_val_valid)]
            episode_history.append(("action", "greedy", action, q_val))
        board, reward, done = self.env.step(action)
        episode_history.append(("reward", reward))
        episode_history.append(("board", board))
        next_featurized = (
            self.model.featurize(board, self.env.my_player) if not done else None
        )
        self.replay_buffer.append((featurized, action, reward, next_featurized))
        return board, next_featurized, done

    def _maybe_train(self):
        """Train the model a few updates if it is time."""
        if (
            self.global_step >= self.learning_starts
            and self.global_step - self.last_train_update >= self.train_freq
        ):
            self.last_train_update = self.global_step
            for _ in range(self.gradient_steps):
                batch = self.replay_buffer.sample(self.batch_size)
                initial_states = maybe_to_cuda(
                    torch.stack([maybe_to_cuda(s) for s, _, _, _ in batch])
                )
                actions = maybe_to_cuda(
                    torch.from_numpy(
                        np.array([a for _, a, _, _ in batch], dtype=np.int64)
                    )
                )
                rewards = maybe_to_cuda(
                    torch.from_numpy(
                        np.array([r for _, _, r, _ in batch], dtype=np.float32)
                    )
                )

                self.model.train()
                self.target_model.eval()
                predicted_q_vals = (
                    self.model(initial_states)
                    .gather(1, actions.unsqueeze(1))
                    .squeeze(1)
                )
                target_q_vals = torch.clone(rewards)
                all_next_states = [s for _, _, _, s in batch]
                if self.target_initialized:
                    idc = maybe_to_cuda(
                        torch.tensor(
                            [i for i, s in enumerate(all_next_states) if s is not None]
                        )
                    )
                    if any(s is not None for s in all_next_states):
                        next_states = maybe_to_cuda(
                            torch.stack(
                                [
                                    maybe_to_cuda(s)
                                    for s in all_next_states
                                    if s is not None
                                ]
                            )
                        )
                        target_q_vals[idc] += (
                            self.gamma
                            * self.target_model(next_states).max(dim=1)[0].detach()
                        )
                # target_q_vals = torch.clamp(target_q_vals, clip_target_q_min, clip_target_q_max)
                unpooled_loss = torch.square(
                    predicted_q_vals - target_q_vals
                )  # .clamp(0, 1)
                loss = unpooled_loss.mean()

                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                self.opt.step()
                self.log_dict(
                    {
                        "loss": loss.detach().cpu().item(),
                        "predicted_q_vals_mean": predicted_q_vals.detach()
                        .cpu()
                        .numpy()
                        .mean(),
                        "target_q_vals_mean": target_q_vals.detach()
                        .cpu()
                        .numpy()
                        .mean(),
                        "rewards_mean": rewards.detach().cpu().numpy().mean(),
                    }
                )
                self._event(
                    "on_training_update",
                    loss=unpooled_loss.detach().cpu(),
                    predicted_q_vals=predicted_q_vals.detach().cpu(),
                    target_q_vals=target_q_vals.detach().cpu(),
                    initial_states=initial_states.detach().cpu(),
                    actions=actions.detach().cpu(),
                    rewards=rewards.detach().cpu(),
                    next_states=all_next_states,
                )

            if (
                self.global_step >= self.target_start
                and self.global_step - self.last_target_update
                >= self.update_target_every
            ):
                print("Update target")
                self.last_target_update = self.global_step
                self.target_model.load_state_dict(
                    copy.deepcopy(self.model.state_dict())
                )
                self.target_initialized = True
                self._event("on_update_target")

    def _train_episode(self):
        """Train the model for a single episode."""
        self._event("on_episode_start")
        board = self.env.reset()
        next_featurized = self.model.featurize(board, self.env.my_player)
        done = False
        episode_history = [("start", board)]
        while not done:
            if (
                self.global_step - self.last_exploration_update
                >= self.exploration_update_freq
            ):
                self.last_exploration_update = self.global_step
                self.epsilon = self.exploration_final_eps + (
                    self.exploration_initial_eps - self.exploration_final_eps
                ) * (1 - (self.global_step / self.max_training_steps))
                self.log_dict({"epsilon": self.epsilon})
                self._event("on_update_epsilon", self.epsilon)
            board, next_featurized, done = self._take_action(
                board, self.epsilon, next_featurized, episode_history
            )
            self._maybe_train()
            self.global_step += 1
        self._event("on_episode_end", episode_history)


class ModelPolicy(Policy):
    """A policy that uses a model to make greedy decisions."""

    def __init__(self, model):
        self.model = model

    def __call__(self, board, player):
        self.model.eval()
        q_val = (
            self.model(maybe_to_cuda(self.model.featurize(board, player).unsqueeze(0)))
            .squeeze(0)
            .cpu()
            .detach()
            .numpy()
        )
        valid_moves = TickTackToeEnvironment.get_valid_actions(board)
        return valid_moves[random_of_max(q_val[valid_moves])]


class DQNAuditCallback(DDQNCallback):
    """Writes game logs for a few games to a log file in the log directory.

    Also logs game statistics for a games against a larger pool of agents
    to the logger.
    """

    def __init__(
        self,
        envs=None,
        filename: Optional[str] = None,
        audit_every=10_000,
        n_episodes=10,
    ):
        self.envs = envs or []
        self.filename = filename or "audit.log"
        self.f = None
        self.audit_every = audit_every
        self.last_audited = 0
        self.n_episodes = n_episodes
        self.arena = AgentArena.default()

    def on_fit_start(self, trainer: "DDQNTrainer"):
        super().on_fit_start(trainer)
        self.f = open(os.path.join(trainer.logdir, self.filename), "w")

    def on_episode_end(self, trainer: "DDQNTrainer", episode_history):
        super().on_episode_end(trainer, episode_history)
        envs = self.envs or [trainer.env]
        if trainer.global_step - self.last_audited >= self.audit_every:
            self.last_audited = trainer.global_step
            self.f.write(f"AUDIT {trainer.global_step}\n")

            stats = self.arena.evaluate(ModelPolicy(trainer.model))
            trainer.log_dict(stats)
            for name, stat in stats.items():
                self.f.write(f"{name}: {stat:.3f}\n")

            for env in envs:
                for _ in range(10):
                    self.f.write(f"new epsiode\n")
                    board = env.reset()
                    self.f.write(f"{env}\n")
                    trainer.model.eval()
                    done = False
                    while not done:
                        featurized = trainer.model.featurize(board, env.my_player)
                        valid_actions = env.get_valid_actions(board)
                        q_val = (
                            trainer.model(maybe_to_cuda(featurized).unsqueeze(0))
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        q_val_valid = q_val[valid_actions]
                        action = valid_actions[random_of_max(q_val_valid)]
                        board, rewards, done = env.step(action)
                        self.f.write(f"{env}\n")
            self.f.write("\n")
            self.f.flush()

    def on_fit_end(self, trainer: "DDQNTrainer"):
        super().on_fit_end(trainer)
        self.f.close()


class ProgBarLoggerCallback(DDQNCallback):
    """Logs training progress to a progress bar along with some metrics."""

    def __init__(self, update_postfix_every=100):
        self.progbar = None
        self.stack = ExitStack()
        self.emas = {}
        self.last_update = 0
        self.update_postfix_every = update_postfix_every
        self.last_postfix_update = 0

    def on_fit_start(self, trainer: "DDQNTrainer"):
        super().on_fit_start(trainer)
        self.progbar = self.stack.enter_context(
            tqdm.tqdm(total=trainer.max_training_steps)
        )

    def log(self, name, value):
        if name not in self.emas:
            self.emas[name] = EMAMeter(0.95)
        self.emas[name].update(value)

    def on_training_update(
        self,
        trainer: "DDQNTrainer",
        loss,
        predicted_q_vals,
        target_q_vals,
        initial_states,
        actions,
        rewards,
        next_states,
    ):
        super().on_training_update(
            trainer,
            loss,
            predicted_q_vals,
            target_q_vals,
            initial_states,
            actions,
            rewards,
            next_states,
        )
        self.log("loss", loss.mean())
        self.log("mean_train_pred_q", predicted_q_vals.mean())
        self.log("mean_train_target_q", target_q_vals.mean())
        self.log("mean_train_reward", rewards.mean())

    def on_update_epsilon(self, trainer: "DDQNTrainer", epsilon):
        return super().on_update_epsilon(trainer, epsilon)
        self.log("epsilon", epsilon)

    def on_episode_end(self, trainer: "DDQNTrainer", episode_history):
        super().on_episode_end(trainer, episode_history)

        rewards = []
        for history in episode_history:
            if history[0] == "reward":
                rewards.append(history[1])
        self.log("mean_cum_rew", sum(rewards))
        trainer.log_dict(
            {
                "mean_cum_reward_train": sum(rewards),
            }
        )

        incr = trainer.global_step - self.last_update
        self.progbar.update(incr)
        self.last_update = trainer.global_step
        if trainer.global_step - self.last_postfix_update >= self.update_postfix_every:
            self.last_postfix_update = trainer.global_step
            self.progbar.set_postfix({k: v.value for k, v in self.emas.items()})

    def on_fit_end(self, trainer: "DDQNTrainer"):
        super().on_fit_end(trainer)
        self.stack.close()


class WandBLogger:
    """Logs training progress to wandb."""

    def __init__(self):
        wandb.init()

    def log_dict(self, d, step):
        wandb.log(d, step=step)


def train():
    model = TransformerTickTackToeModel()  # FFTickTackToeModel()
    train_agent = lambda: MultiPolicy(
        [
            OptimalPolicy(),
            RandomPolicy(),
        ],
        np.random.dirichlet([3, 2]),
    )
    DDQNTrainer(
        env=TickTackToeEnvironment(train_agent),
        model=model,
        opt=torch.optim.Adam(
            model.parameters(),
            lr=1e-3,  # 5e-4, # 0.00005,
            weight_decay=1e-5,
            betas=(0.8, 0.8),
        ),
        gradient_clipping=1.0,
        learning_starts=1000,
        target_start=5_000,
        update_target_every=1_000,
        max_training_steps=1_000_000,
        callbacks=[
            ProgBarLoggerCallback(),
            DQNAuditCallback(
                envs=[
                    TickTackToeEnvironment(train_agent),
                    TickTackToeEnvironment(OptimalPolicy),
                ]
            ),
        ],
        logger=WandBLogger(),
    ).fit()


if __name__ == "__main__":
    train()
