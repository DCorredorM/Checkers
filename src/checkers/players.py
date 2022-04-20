from abc import ABC, abstractmethod
from checkers.board import StateVector, StateTransitions
import random as rnd
from checkers.function_approximation import *
import numpy as np


class CheckersPlayer(ABC):
    """
    Abstract class that represents a checkers player.

    Every instance needs to implement the `next_move` method that defines its policy.
    """
    def __init__(self):
        pass

    @abstractmethod
    def next_move(self, state: StateVector) -> StateVector:
        """
        Abstract method that will define each player's strategy.

        Parameters
        ----------
        state: StateVector
            The state in which the player needs to take the move.

        Returns
        -------
        StateVector:
            The state in which the game will be after the move of the player.
        """
        ...


class UniformPlayer(CheckersPlayer):
    """Represents a player whose strategy is to choose randomly among the feasible moves."""
    def next_move(self, state: StateVector) -> StateVector:
        """
        Chooses a random uniform move given the state.

        Parameters
        ----------
        state: StateVector
            The state in which the player needs to take the move.

        Returns
        -------
        StateVector:
            The state in which the game will be after the move of the player.
        """
        next_moves = StateTransitions.feasible_next_moves(state)
        if next_moves:
            return rnd.choice(next_moves)


class EpsilonGreedyPlayer(CheckersPlayer):
    DEFAULT_EPSILON = 0.1
    
    def __init__(self, q_approximation: BaseQApproximation, epsilon=None):
        super().__init__()
        self.epsilon = EpsilonGreedyPlayer.DEFAULT_EPSILON if epsilon is None else epsilon
        self.q = q_approximation
    
    def next_move(self, state: StateVector) -> StateVector:
        next_moves = {hash(s): s for s in StateTransitions.feasible_next_moves(state)}
        if next_moves:
            if np.random.rand() < 1-self.epsilon:
                q_factors = {k: self.q((state, action)) for k, action in next_moves.items()}
                
                return next_moves[max(q_factors.keys(), key=lambda x: q_factors[x])]
            else:
                return rnd.choice(list(next_moves.values()))
