import numpy as np

from checkers.environment import StateVector
from checkers.players import UniformPlayer, CheckersPlayer
from typing import Optional, List, Union, Tuple


class CheckersGame:
	"""
	Represents a checkers game.

	Attributes:
	__________
	red_player: CheckersPlayer
		The red player.
	blue_player: CheckersPlayer
		The blue player.
	"""
	def __init__(self, red_player: CheckersPlayer, blue_player: CheckersPlayer):
		self.red_player = red_player
		self.blue_player = blue_player

	def simulate_game(
			self,
			number_of_moves: Optional[int] = np.inf,
			initial_state: Optional[StateVector] = None
	) -> Tuple[List[StateVector], int]:
		"""
		Simulate one game of checkers starting in initial_state and with maximum number of moves of number_of_moves.

		Parameters
		----------
		number_of_moves: Optional[int]
			The maximum number of turns that want to be simulated.
		initial_state: Optional[StateVector]
			The initial state in which the simulation of the game will start.

		Returns
		-------
		Tuple[List[StateVector], int]
			A list of StateVector objects representing the history of the game, and a boolean value that indicates which
			player won the game. 1 means that the blue player won and zero that the red player won.
			# Todo: Check convention!!
		"""
		if initial_state is None:
			initial_state = StateVector()

		history = []
		winner = None

		current_state = initial_state
		turn = 0
		while True:
			history.append(current_state)
			player = self.blue_player if turn % 2 == 0 else self.red_player
			next_state = player.next_move()

			# if the game is over (i.e., current player has no moves)
			# or the turn exceeds the maximum number of turns allowed the iteration is stoped.
			if next_state is None or turn > number_of_moves:
				winner = turn % 2 == 0
				break
			else:
				current_state = next_state
			turn += 1

		return history, winner

	def simulate_games(self, number_of_games: int) -> Tuple[List[List[StateVector]], List[int]]:
		"""
		Simulate number_of_games checkers games until someone wins.

		Parameters
		----------
		number_of_games: int
			Number of games that will be played.

		Returns
		-------
		Tuple[List[List[StateVector]], List[int]]
			The list of histories and the list of results for the simulated games.
		"""
		histories = []
		results = []

		for i in range(number_of_games):
			history, result = self.simulate_game()
			histories.append(history)
			results.append(result)

		return histories, results
