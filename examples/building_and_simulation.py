from checkers import (
    StateVector,
    StateTransitions,
    CheckersGame,
    UniformPlayer,
    EpsilonGreedyPlayer,
    MaterialBalanceApprox
)
import os

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time


def create_game():
    epsilon = 0.1
    q_approx = MaterialBalanceApprox()
    
    game = CheckersGame(
        light_player=EpsilonGreedyPlayer(q_approx, epsilon=epsilon),
        dark_player=UniformPlayer()
    )
    return game


def simulate_visual(portion=None):
    game = create_game()

    history, winner = game.simulate_game()
    path = os.path.join('data', 'games', f'{time.time()}')
    os.makedirs(path, exist_ok=True)
    print(f'the winner is {winner}')
    if portion is None:
        i = 0
        for s in history:
            s.visualize(save_path=os.path.join(path, f'frame{i}.png'), show=False)
            i += 1
    else:
        for s in history[-portion:]:
            s.visualize()


def simulate():
    game = create_game()
    history, winner = game.simulate_game()
    return history, winner


if __name__ == '__main__':
    seed = 754
    np.random.seed(seed)
    rnd.seed(seed)
    simulate()
