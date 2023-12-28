import datetime
from typing import Optional
import numpy as np
import torch
import copy
import tqdm
import torch.nn.functional as F
import os
from utils import EMAMeter, maybe_to_cuda, random_of_max
from replay_buffer import CircularBuffer
from contextlib import ExitStack
from ttt_env import (
    Policy,
    TickTackToeEnvironment,
    MultiPolicy,
    RandomPolicy,
    OptimalPolicy,
)
from models import FFTickTackToeModel, TransformerTickTackToeModel
from arena import AgentArena
import wandb


class DQNCallback:
    def on_fit_start(self, trainer: "DQNTrainer"):
        ...

    def on_episode_start(self, trainer: "DQNTrainer"):
        ...

    def on_training_update(
        self,
        trainer: "DQNTrainer",
        loss,
        predicted_q_vals,
        target_q_vals,
        initial_states,
        actions,
        rewards,
        next_states,
    ):
        ...

    def on_update_target(self, trainer: "DQNTrainer"):
        ...

    def on_episode_end(self, trainer: "DQNTrainer", episode_history):
        ...

    def on_update_epsilon(self, trainer: "DQNTrainer", epsilon):
        ...

    def on_fit_end(self, trainer: "DQNTrainer"):
        ...


class DDQNTrainer:
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
        callbacks: list[DQNCallback] = [],
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

    def log_dict(self, d):
        if self.logger:
            self.logger.log_dict(d, self.global_step)

    def _event(self, event_name, *args, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event_name)(self, *args, **kwargs)

    def fit(self):
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


class DQNAuditCallback(DQNCallback):
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

    def on_fit_start(self, trainer: "DQNTrainer"):
        super().on_fit_start(trainer)
        self.f = open(os.path.join(trainer.logdir, self.filename), "w")

    def on_episode_end(self, trainer: "DQNTrainer", episode_history):
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

    def on_fit_end(self, trainer: "DQNTrainer"):
        super().on_fit_end(trainer)
        self.f.close()


class ProgBarLoggerCallback(DQNCallback):
    def __init__(self, update_postfix_every=100):
        self.progbar = None
        self.stack = ExitStack()
        self.emas = {}
        self.last_update = 0
        self.update_postfix_every = update_postfix_every
        self.last_postfix_update = 0

    def on_fit_start(self, trainer: "DQNTrainer"):
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
        trainer: "DQNTrainer",
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

    def on_update_epsilon(self, trainer: "DQNTrainer", epsilon):
        return super().on_update_epsilon(trainer, epsilon)
        self.log("epsilon", epsilon)

    def on_episode_end(self, trainer: "DQNTrainer", episode_history):
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

    def on_fit_end(self, trainer: "DQNTrainer"):
        super().on_fit_end(trainer)
        self.stack.close()


class WandBLogger:
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

    # model = maybe_to_cuda(model)
    # target_model = maybe_to_cuda(target_model)

    # opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # replay_buffer = CircularBuffer(replay_buffer_len)
    # global_step = 0
    # postfix = {
    #     'loss': EMAMeter(0.95),
    #     'valid_pct': EMAMeter(0.95),
    #     'mean_rew': EMAMeter(0.95),
    #     'mean_q': EMAMeter(0.95),
    #     'eps': EMAMeter(0.95),
    # }
    # last_audited = 0
    # last_target_update = 0
    # with open('training.log', 'w') as f:
    #     with tqdm.tqdm(total=max_training_steps) as pbar:
    #         epsilon = exploration_initail_eps
    #         my_player = np.random.randint(2)
    #         while global_step < max_training_steps:
    #             board, valid_actions = env.reset()
    #             done = False
    #             player = 0
    #             pending = None
    #             ep_valid_moves = []
    #             ep_rewards = []
    #             ep_loss = []
    #             ep_epsilon = []
    #             ep_q = []
    #             ep_t = 0
    #             while not done:
    #                 featurized = featurize(board, player)
    #                 if global_step % exploration_steps == 0:
    #                     epsilon = exploration_final_eps + (exploration_initail_eps - exploration_final_eps) * np.exp(-global_step / max_training_steps)
    #                 ep_epsilon.append(epsilon)
    #                 assert len(valid_actions) > 0
    #                 if player == my_player:
    #                     if np.random.rand() < epsilon:
    #                         action = valid_actions[np.random.randint(len(valid_actions))]
    #                     else:
    #                         model.eval()
    #                         q_val = model(maybe_to_cuda(featurized).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    #                         q_val_valid = q_val[valid_actions]
    #                         # check_soft_divergence(q_val_valid, gamma)
    #                         action = valid_actions[random_of_max(q_val_valid)]
    #                         ep_q.append(q_val[action].item())
    #                 else:
    #                     action = oracle(board, player)
    #                 board, rewards, done, was_valid, valid_actions = env.step(action)
    #                 #next_featurized = featurize(board, player)
    #                 player = (player + 1) % 2
    #                 if player == my_player:
    #                     ep_valid_moves.append(was_valid)
    #                     for rew in rewards:
    #                         if rew is not None:
    #                             ep_rewards.append(rew)
    #                 my_reward, their_reward = rewards
    #                 # gamma = player_t_to_gamma[ep_t // 2]
    #                 ep_t += 1
    #                 if pending is not None:
    #                     # Update reward for other player if they lost
    #                     pending_state, pending_action, pending_reward = pending
    #                     next_featurized = None if done else featurize(board, player)
    #                     replay_buffer.append((pending_state, pending_action, their_reward if done else pending_reward, next_featurized, False))
    #                 pending = (featurized, action, my_reward)
    #                 if done and pending:
    #                     # Finished episode; flush pending immediately
    #                     pending_state, pending_action, pending_reward = pending
    #                     replay_buffer.append((pending_state, pending_action, pending_reward, None, True))
    #                     pending = None
    #                 if global_step >= learning_starts and global_step % train_freq == 0:
    #                     for gradient_step in range(gradient_steps):
    #                         batch = replay_buffer.sample(batch_size)
    #                         initial_states = maybe_to_cuda(torch.stack([maybe_to_cuda(s) for s, _, _, _, _ in batch]))
    #                         actions = maybe_to_cuda(torch.from_numpy(np.array([a for _, a, _, _, _ in batch], dtype=np.int64)))
    #                         rewards = maybe_to_cuda(torch.from_numpy(np.array([r for _, _, r, _, _ in batch], dtype=np.float32)))
    #                         is_done = maybe_to_cuda(torch.from_numpy(np.array([1.0 if d else 0.0 for _, _, _, _, d in batch], dtype=np.float32)))
    #                         # gammas = maybe_to_cuda(torch.from_numpy(np.array([g for _, _, _, _, g in batch], dtype=np.float32)))

    #                         model.train()
    #                         target_model.eval()
    #                         predicted_q_vals = model(initial_states).gather(1, actions.unsqueeze(1)).squeeze(1)
    #                         target_q_vals = rewards
    #                         if target_initialized:
    #                             all_next_states = [s for _, _, _, s, _ in batch]
    #                             idc = maybe_to_cuda(torch.tensor([i for i, s in enumerate(all_next_states) if s is not None]))
    #                             if any(s is not None for s in all_next_states):
    #                                 next_states = maybe_to_cuda(torch.stack([maybe_to_cuda(s) for s in all_next_states if s is not None]))
    #                                 target_q_vals[idc] += gamma * target_model(next_states).max(dim=1)[0].detach()
    #                         # target_q_vals = torch.clamp(target_q_vals, clip_target_q_min, clip_target_q_max)
    #                         loss = torch.square(predicted_q_vals - target_q_vals).clamp(0, 1).mean()

    #                         loss.backward()
    #                         opt.step()

    #                     if global_step >= target_start and global_step - last_target_update >= update_target_every:
    #                         print("Update target")
    #                         last_target_update = global_step
    #                         target_model.load_state_dict(copy.deepcopy(model.state_dict()))
    #                         target_initialized = True

    #                     ep_loss.append(loss.item())
    #                     pbar.update(1)
    #                 else:
    #                     pbar.update(1)

    #                 global_step += 1

    #             if global_step - last_audited >= 10_000:
    #                 print("Audit")
    #                 last_audited = global_step
    #                 f.write(f'AUDIT {global_step}\n')
    #                 for _ in range(10):
    #                     f.write(f'new epsiode\n')
    #                     my_player = np.random.randint(2)
    #                     f.write(f'my_player: {my_player}\n')
    #                     board, valid_actions = env.reset()
    #                     player = 0
    #                     model.eval()
    #                     target_model.eval()
    #                     pending = None
    #                     done = False
    #                     ep_t = 0
    #                     while not done:
    #                         featurized = featurize(board, player)
    #                         if player == my_player:
    #                             q_val = model(maybe_to_cuda(featurized).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    #                             q_val_valid = q_val[valid_actions]
    #                             action = valid_actions[random_of_max(q_val_valid)]
    #                         else:
    #                             action = oracle(board, player)
    #                         board, rewards, done, was_valid, valid_actions = env.step(action)
    #                         next_featurized = featurize(board, player)
    #                         # gamma = player_t_to_gamma[ep_t // 2]
    #                         ep_t += 1

    #                         def flush_pending(include_extra=0):
    #                             if pending:
    #                                 pending_q, pending_action, pending_reward, pending_next, gamma = pending
    #                                 if include_extra:
    #                                     pending_reward = include_extra
    #                                 target_q_next= target_model(maybe_to_cuda(pending_next.unsqueeze(0))).detach().cpu()
    #                                 target_q_val = pending_reward + gamma * target_q_next.max(dim=1)[0]
    #                                 f.write(f'action:{pending_action}, reward:{pending_reward}, q_val:{pending_q.tolist()}, target_q_val:{target_q_val}\n')
    #                         if pending is not None:
    #                             # Update reward for other player if they lost
    #                             flush_pending(rewards[1] or 0)
    #                         pending = (q_val, action, rewards[0], next_featurized, gamma) if player == my_player else None
    #                         if done:
    #                             flush_pending()
    #                         if player != my_player:
    #                             f.write(f'oracle action:{action}\n')
    #                         player = (player + 1) % 2
    #                 f.write('\n')
    #                 f.flush()

    #             postfix['mean_rew'].update(np.mean(ep_rewards))
    #             if ep_q:
    #                 postfix['mean_q'].update(np.mean(ep_q))
    #             postfix['valid_pct'].update(np.mean([1.0 if valid else 0.0 for valid in ep_valid_moves]))
    #             postfix['eps'].update(np.mean(ep_epsilon))
    #             if ep_loss:
    #                 postfix['loss'].update(np.mean(ep_loss))
    #             if global_step % 1000 == 0:
    #                 pbar.set_postfix({k: v.value for k, v in postfix.items()})


if __name__ == "__main__":
    train()
    # env = TickTackToeEnvironment()
    # board = env.reset()
    # env.render()
    # env.step(4)
    # env.render()
    # env.step(0)
    # env.render()
    # env.step(6)
    # env.render()
