from checkers import (
    StateVector,
    StateTransitions,
    CheckersGame,
    UniformPlayer,
    EpsilonGreedyPlayer,
    MaterialBalanceApprox,
    AlphaBetaPlayer,
    EpsilonAlphaBetaPlayer
)

from gui import Visualizer

import os

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time


def create_game():
    epsilon = 0.1
    v_approx = MaterialBalanceApprox()
    
    # game = CheckersGame(
    #     light_player=AlphaBetaPlayer(v_approx, depth=4),
    #     dark_player=AlphaBetaPlayer(v_approx, depth=4)
    # )
    game = CheckersGame(
        light_player=AlphaBetaPlayer(v_approx, depth=4),
        dark_player=UniformPlayer()
    )
    return game


def simulate_visual(portion=None):
    game = create_game()

    history, winner = game.simulate_game()
    print(f'the winner is {winner}')
    path = os.path.join('data', 'games', f'{game}-{time.strftime("%b %d %Y %H:%M:%S")}')
    
    Visualizer.visualize_game(history, path)


def simulate():
    game = create_game()
    history, winner = game.simulate_game()
    return history, winner


def create_data():
    game = create_game()
    game.td_lambda_training(
        value_function=game.light_player.value_function,
        number_of_games=10,
        gamma=0.9,
        lambda_=0.9,
        train=False,
    )


if __name__ == '__main__':
    seed = 754
    np.random.seed(seed)
    rnd.seed(seed)
    simulate_visual()
