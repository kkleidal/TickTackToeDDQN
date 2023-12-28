import gzip
import os
import pickle
import subprocess
from functools import cache
from typing import Optional, Tuple

import numpy as np


class Policy:
    def __call__(self, board, player):
        raise NotImplementedError


class OptimalPolicy(Policy):
    def __init__(self, deterministic=False):
        self.deterministic = deterministic
        self.policy = OptimalPolicy._load_optimal_policy()

    @staticmethod
    @cache
    def _load_optimal_policy():
        self_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(os.path.join(self_dir, "optimal_policy.pkl.gz")):
            subprocess.run(
                ["python", "compute_optimal_strategy.py"], cwd=self_dir, check=True
            )
        assert os.path.exists(os.path.join(self_dir, "optimal_policy.pkl.gz"))
        with gzip.open("optimal_policy.pkl.gz", "rb") as f:
            policy = pickle.load(f)
        return policy

    def __call__(self, board, player):
        board_tuple = tuple(tuple(row) for row in board)
        assert player in (0, 1)
        options = self.policy[(board_tuple, player)]
        assert len(options) > 0
        if self.deterministic:
            return options[0]
        else:
            return np.random.choice(options)


class RandomPolicy(Policy):
    def __call__(self, board, player):
        return np.random.choice(np.nonzero(board.flatten() == 0)[0])


# class Dist:
#     def sample(self, n):
#         raise NotImplementedError
#
# class ConstantProbabilityDist(Dist):
#     def __init__(self, probabilities):
#         self.probabilities = probabilities
#
#     def sample(self, n):
#         assert n == len(self.probabilities)
#         return self.probabilities
#
# class DirichletProbabilityDist(Dist):
#     def __init__(self, weights=None):
#         self.weights = weights
#
#     def sample(self, n):
#         assert self.weights is None or len(self.weights) == n
#         return np.random.dirichlet(np.ones(n) if self.weights is None else self.weights)


class MultiPolicy(Policy):
    def __init__(self, policies, probabilities):
        self.policies = policies
        self.probabilities = probabilities

    def __call__(self, board, player):
        return self.policies[
            np.random.choice(len(self.policies), p=self.probabilities)
        ](board, player)


ComboReward = Tuple[Optional[float], Optional[float]]

INVALID_MOVE_REWARD = -1
VALID_MOVE_REWARD = -0.01
WIN_REWARD = 1
LOSS_REWARD = -1
TIE_REWARD = 0

WIN = 2
TIE = 1
UNRESOLVED = 0
LOSS = -1


def three_in_a_row(m):
    return (
        np.any(np.count_nonzero(m, 1) == 3)
        or np.any(np.count_nonzero(m, 0) == 3)
        or np.count_nonzero(np.diag(m)) == 3
        or np.count_nonzero(np.diag(np.rot90(m))) == 3
    )


class TickTackToeEnvironment:
    def __init__(
        self,
        opponent_fn=OptimalPolicy,
        invalid_move_reward=INVALID_MOVE_REWARD,
        valid_move_reward=VALID_MOVE_REWARD,
        win_reward=WIN_REWARD,
        loss_reward=LOSS_REWARD,
        tie_reward=TIE_REWARD,
    ):
        self.opponent_fn = opponent_fn

        self.board = np.zeros([3, 3], dtype=np.uint8)
        self.turn = 0
        self.my_player = 0

        self.opponent = None
        self.invalid_move_reward = invalid_move_reward
        self.valid_move_reward = valid_move_reward
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.tie_reward = tie_reward

    @staticmethod
    def get_valid_actions(board):
        return np.nonzero(board.flatten() == 0)[0]

    def reset(self):
        self.opponent = self.opponent_fn()
        self.board = np.zeros([3, 3], dtype=np.uint8)
        self.turn = 0
        self.my_player = np.random.randint(2)
        board = self.board
        while self.my_player != self.turn:
            board, done = self._only_step(self.opponent(board, self.turn))
            assert not done
        return board

    def _get_state(self):
        p1_spots = self.board == 1
        p2_spots = self.board == 2
        if three_in_a_row(p1_spots):
            return WIN if self.my_player == 0 else LOSS
        elif three_in_a_row(p2_spots):
            return WIN if self.my_player == 1 else LOSS
        elif np.all(self.board != 0):
            return TIE
        else:
            return UNRESOLVED

    # def get_valid_actions(self):
    #     return np.nonzero(self.board.flatten() == 0)[0]

    def _only_step(self, action) -> Tuple[np.ndarray, bool]:
        if action >= 9 or action < 0:
            raise ValueError
        y = action // 3
        x = action % 3
        if self.board[y, x] != 0:
            return self.board, False
        player = self.turn + 1
        self.turn = (self.turn + 1) % 2
        self.board[y, x] = player
        state = self._get_state()
        if state == WIN:
            return self.board, True
        elif state == LOSS:
            return self.board, True
        elif state == TIE:
            return self.board, True
        elif state == UNRESOLVED:
            return self.board, False

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        if action >= 9 or action < 0:
            raise ValueError
        y = action // 3
        x = action % 3
        if self.board[y, x] != 0:
            return self.board, self.invalid_move_reward, False
        assert self.turn == self.my_player
        player = self.turn + 1
        self.turn = (self.turn + 1) % 2
        self.board[y, x] = player
        state = self._get_state()
        if state == WIN:
            return self.board, self.win_reward, True
        elif state == TIE:
            return self.board, self.tie_reward, True
        elif state == UNRESOLVED:
            self._only_step(self.opponent(self.board, self.turn))
            state = self._get_state()
            if state == LOSS:
                return self.board, self.loss_reward, True
            elif state == TIE:
                return self.board, self.tie_reward, True
            elif state == UNRESOLVED:
                return self.board, self.valid_move_reward, False
            else:
                assert False
        else:
            assert False

    def __str__(self):
        return (
            f'My player: {"X" if self.my_player == 0 else "O"}\n'
            + "\n".join(
                "".join(
                    {
                        0: ".",
                        1: "X",
                        2: "O",
                    }[c]
                    for c in row
                )
                for row in self.board.tolist()
            )
            + "\n"
        )

    def render(self, mode="human"):
        print(str(self))
