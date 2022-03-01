from checkers import StateVector, StateTransitions, CheckersGame, UniformPlayer

import numpy as np
import matplotlib.pyplot as plt
import random as rnd


def simulate_visual(portion=None):
	game = CheckersGame(
		light_player=UniformPlayer(),
		dark_player=UniformPlayer()
	)

	history, winner = game.simulate_game()

	print(f'the winner is {winner}')
	if portion is None:
		i = 0
		for s in history:
			s.visualize(save_path=f'data/pictures/frame{i}.png', show=False)
			i += 1
	else:
		for s in history[-portion:]:
			s.visualize()


def simulate():
	game = CheckersGame(
		light_player=UniformPlayer(),
		dark_player=UniformPlayer()
	)

	history, winner = game.simulate_game()
	return history, winner


if __name__ == '__main__':
	seed = 756
	np.random.seed(seed)
	rnd.seed(seed)
	simulate_visual()
