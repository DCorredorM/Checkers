from checkers import (
    StateVector,
    StateTransitions,
    CheckersGym,
    UniformPlayer,
    EpsilonGreedyPlayer,
    MaterialBalanceApprox,
    AlphaBetaPlayer,
    EpsilonAlphaBetaPlayer, NNetApprox, DNNApprox
)

from gui import Visualizer

import os

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time

from utils.data_handler import load_training_data


def create_game():
    epsilon = 0.1
    v_approx = MaterialBalanceApprox()
    
    game = CheckersGym(
        light_player=AlphaBetaPlayer(v_approx, depth=4),
        dark_player=AlphaBetaPlayer(v_approx, depth=6)
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


def test_nnet():
    dataset = 'UniformPlayer(-1)vsAlphaBetaPlayer(1)(4)'
    states, targets = load_training_data(
        os.path.join('data', 'training_data', dataset)
    )

    path = os.path.join('data', 'models', 'power_net')
    structure = [512] * 3 + [256] * 3
    
    # nnet = DNNApprox(structure=structure)
    nnet = DNNApprox.load(path)
    y_hat = nnet(states[0])
    
    for e in range(1000):
        loss = nnet.train_batch(states, targets)
        print(loss)
    
    y_hat_2 = nnet(states)
    
    nnet.save(path)
    print('')
    
    
def main():
    # seed = 754
    # np.random.seed(seed)
    # rnd.seed(seed)
    # simulate_visual()
    test_nnet()


if __name__ == '__main__':
    main()

