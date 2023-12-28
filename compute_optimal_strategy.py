import gzip
import pickle
from functools import cache

import numpy as np


def three_in_a_row(m):
    return (
        np.any(np.count_nonzero(m, 1) == 3)
        or np.any(np.count_nonzero(m, 0) == 3)
        or np.count_nonzero(np.diag(m)) == 3
        or np.count_nonzero(np.diag(np.rot90(m))) == 3
    )


def board_to_numpy(board):
    return np.array(list(list(row) for row in board))


def board_to_tuple(board):
    return tuple(tuple(row) for row in board)


possible_states = set()


@cache
def optimal_strategy(board, player):
    possible_states.add((board, player))
    board_np = board_to_numpy(board)
    if three_in_a_row(board_np == (player + 1)):
        return ((1, -1), [])
    if three_in_a_row(board_np == (((player + 1) % 2) + 1)):
        return ((-1, 1), [])
    if np.count_nonzero(board_np) == 9:
        return ((0, 0), [])
    possible = []
    for i in np.nonzero(board_np.flatten() == 0)[0]:
        board2 = board_np.copy()
        board2[i // 3, i % 3] = player + 1
        value, _ = optimal_strategy(board_to_tuple(board2), (player + 1) % 2)
        possible.append(((value[1], value[0]), i))
    best = min(possible, key=lambda x: (x[0][1], -x[0][0]))
    value = best[0]
    return value, [x[1] for x in possible if x[0] == value]


init_state = board_to_tuple(np.zeros((3, 3), dtype=np.int32))
print(optimal_strategy(init_state, 0))
print(len(possible_states))
policy = {state: optimal_strategy(*state)[1] for state in possible_states}
with gzip.open("optimal_policy.pkl.gz", "wb") as f:
    pickle.dump(policy, f)
