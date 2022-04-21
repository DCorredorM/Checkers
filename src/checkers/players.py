from abc import ABC, abstractmethod
from checkers.board import StateVector, StateTransitions
from checkers.piece import PieceHelper
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
    
    def __init__(self, value_function_approximation: BaseJApproximation, epsilon=None):
        super().__init__()
        self.epsilon = EpsilonGreedyPlayer.DEFAULT_EPSILON if epsilon is None else epsilon
        self.value_function = value_function_approximation
    
    def next_move(self, state: StateVector) -> StateVector:
        next_moves = {hash(s): s for s in StateTransitions.feasible_next_moves(state)}
        if next_moves:
            if np.random.rand() < 1-self.epsilon:
                q_factors = {k: self.value_function.q_factor(state, action) for k, action in next_moves.items()}
                
                return next_moves[max(q_factors.keys(), key=lambda x: q_factors[x])]
            else:
                return rnd.choice(list(next_moves.values()))


class AlphaBetaPlayer(CheckersPlayer):
    DEFAULT_DEPTH = 3
    
    def __init__(self, value_function_approximation: BaseJApproximation, depth=None):
        super().__init__()
        self.value_function = value_function_approximation
        self.depth = AlphaBetaPlayer.DEFAULT_DEPTH if depth is None else depth
    
    def next_move(self, state: StateVector) -> StateVector:
        val, state = self.minimax(state, -np.inf, np.inf, self.depth)
        return state
    
    def minimax(self, state: StateVector, alpha: float, beta: float, depth: int):
        if depth == 0 or state.is_final():
            return self.value_function(state), None
        next_state = None
        if state.turn == PieceHelper.light:
            # maximizing player turn
            max_eval = -np.inf
            for s in StateTransitions.feasible_next_moves(state):
                cur_eval, next_state = self.minimax(s, alpha, beta, depth-1)
                max_eval = max(max_eval, cur_eval)
                if max_eval == cur_eval:
                    next_state = s
                alpha = max(alpha, cur_eval)
                if beta <= alpha:
                    break
            return max_eval, next_state
        else:
            # minimizing player turn
            min_eval = np.inf
            for s in StateTransitions.feasible_next_moves(state):
                cur_eval, next_state = self.minimax(s, alpha, beta, depth - 1)
                min_eval = min(min_eval, cur_eval)
                if min_eval == cur_eval:
                    next_state = s
                beta = min(beta, cur_eval)
                if beta <= alpha:
                    break
            return min_eval, next_state
            
        