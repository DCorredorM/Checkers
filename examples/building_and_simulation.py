from checkers import (
    StateVector,
    StateTransitions,
    CheckersGym,
    UniformPlayer,
    EpsilonGreedyPlayer,
    MaterialBalanceApprox,
    AlphaBetaPlayer,
    EpsilonAlphaBetaPlayer,
    NNetApprox,
    PieceHelper
)

from gui import Visualizer

import os

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time

from utils.data_handler import load_training_data


def create_game():
    epsilon = 0.4
    v_approx = MaterialBalanceApprox()
    
    game = CheckersGym(
        light_player=EpsilonGreedyPlayer(v_approx, color=PieceHelper.light, depth=4, epsilon=epsilon),
        dark_player=AlphaBetaPlayer(v_approx, color=PieceHelper.dark, depth=6, epsilon=epsilon)
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
        number_of_games=20,
        gamma=0.9,
        lambda_=0.9,
        train=False,
    )

    
def main():
    # seed = 754
    # np.random.seed(seed)
    # rnd.seed(seed)
    # simulate_visual()
    create_data()


if __name__ == '__main__':
    main()

