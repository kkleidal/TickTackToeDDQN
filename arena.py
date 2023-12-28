from typing import Dict, List

import numpy as np

from ttt_env import MultiPolicy, OptimalPolicy, Policy, RandomPolicy, three_in_a_row


def do_match(agent1, agent2):
    board = np.zeros((3, 3), dtype=np.int32)
    player = 0
    while True:
        action = agent1(board, player) if player == 0 else agent2(board, player)
        board[action // 3, action % 3] = player + 1
        if three_in_a_row(board == player + 1):
            return player
        if np.all(board != 0):
            return -1
        player = 1 - player


class AgentArena:
    def __init__(self, agents: Dict[str, Policy]):
        self.agents = agents

    def evaluate(self, agent, trials=100):
        assert trials % 2 == 0
        stats = {}
        for opp_name, opponent in self.agents.items():
            wins = 0
            losses = 0
            ties = 0
            for _ in range(trials // 2):
                outcome = do_match(agent, opponent)
                if outcome == 0:
                    wins += 1
                elif outcome == 1:
                    losses += 1
                elif outcome == -1:
                    ties += 1
            for _ in range(trials // 2):
                outcome = do_match(opponent, agent)
                if outcome == 0:
                    losses += 1
                elif outcome == 1:
                    wins += 1
                elif outcome == -1:
                    ties += 1
            stats[f"{opp_name}/win_rate"] = wins / (wins + losses + ties)
            stats[f"{opp_name}/win_or_tie_rate"] = (wins + ties) / (
                wins + losses + ties
            )
        return stats

    @classmethod
    def default(cls):
        return cls(
            {
                **{
                    "opteps%.1f"
                    % (1 - i): MultiPolicy(
                        [
                            OptimalPolicy(),
                            RandomPolicy(),
                        ],
                        [i, 1 - i],
                    )
                    for i in np.arange(0.1, 1.01, 0.1)
                },
                "opt": OptimalPolicy(),
                "rand": RandomPolicy(),
            }
        )


if __name__ == "__main__":
    arena = AgentArena(
        {
            **{
                "opteps%.1f"
                % (1 - i): MultiPolicy(
                    [
                        OptimalPolicy(),
                        RandomPolicy(),
                    ],
                    [i, 1 - i],
                )
                for i in np.arange(0.1, 1.01, 0.1)
            },
            "opt": OptimalPolicy(),
            "rand": RandomPolicy(),
        }
    )
    print(
        arena.evaluate(
            MultiPolicy(
                [
                    OptimalPolicy(),
                    RandomPolicy(),
                ],
                [0.5, 0.5],
            )
        )
    )
