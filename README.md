## Tick Tack Toe Deep-Q Learning

Even though tick tack toe is an easily solved game (as evidenced by compute_optimal_strategy.py),
I wanted a very simple game to try to implement a deep-Q learning agent from scratch as an exercise.

```
python train.py
```
will train the model and write logs to wandb. It trains by playing against various epsilon-greedy agents
playing the optimal policy stochastically (both in terms of which of multiple possible optimal moves to play,
and in terms of whether it makes the optimal move or a blunder -- the epsilon).
