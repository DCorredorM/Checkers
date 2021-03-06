from abc import ABC, abstractmethod
from copy import copy
from typing import Optional

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
    COLOR = PieceHelper.light
    
    def __init__(self, value_function_approximation: BaseJApproximation = None, color=None, **kwargs):
        if color is None:
            self.color = AlphaBetaPlayer.COLOR
            AlphaBetaPlayer.COLOR ^= PieceHelper.toggle_turn
        else:
            self.color = color
        self.value_function: Optional[BaseJApproximation] = MaterialBalanceApprox() \
            if value_function_approximation is None else value_function_approximation

    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    @abstractmethod
    def next_move(self, state: StateVector, value_func_approx=None) -> StateVector:
        """
        Abstract method that will define each player's strategy.

        Parameters
        ----------
        value_func_approx: list
            List that records the new estimate of the value function at the given state.
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
    def next_move(self, state: StateVector, value_func_approx=None) -> StateVector:
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
        if value_func_approx is None:
            value_func_approx = []
            
        next_moves = StateTransitions.feasible_next_moves(state)
        if next_moves:
            next_move = rnd.choice(next_moves)
            value_func_approx.append(self.value_function.q_factor(state, next_move))
            return next_move


class EpsilonGreedyPlayer(CheckersPlayer):
    DEFAULT_EPSILON = 0.1
    
    def __init__(self, value_function_approximation: BaseJApproximation, epsilon=None, color=None, **kwargs):
        super().__init__(color)
        self.epsilon = EpsilonGreedyPlayer.DEFAULT_EPSILON if epsilon is None else epsilon
        self.value_function = value_function_approximation
    
    def next_move(self, state: StateVector, value_func_approx=None) -> StateVector:
        if value_func_approx is None:
            value_func_approx = []
        
        if not state.is_final():
            next_moves = {hash(s): s for s in StateTransitions.feasible_next_moves(state)}
            if np.random.rand() < 1-self.epsilon:
                q_factors = {k: self.value_function.q_factor(state, action) for k, action in next_moves.items()}
                next_move = next_moves[max(q_factors.keys(), key=lambda x: q_factors[x])]
                value_func_approx.append(self.value_function.q_factor(state, next_move))
                return next_move
            else:
                next_move = rnd.choice(list(next_moves.values()))
                value_func_approx.append(self.value_function.q_factor(state, next_move))
                return next_move
        else:
            return state


class AlphaBetaPlayer(CheckersPlayer):
    DEFAULT_DEPTH = 3
    
    def __init__(self,
                 value_function_approximation: BaseJApproximation,
                 color=None,
                 depth=None,
                 **kwargs):
        super().__init__(color)
        self.value_function = value_function_approximation
        self.depth = AlphaBetaPlayer.DEFAULT_DEPTH if depth is None else depth
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.depth})'
    
    def next_move(self, state: StateVector, value_func_approx=None) -> StateVector:
        if value_func_approx is None:
            value_func_approx = []
        val, next_state = self.minimax(state, -np.inf, np.inf, self.depth)
        value_func_approx.append(val)
        return next_state
    
    def minimax(self, state: StateVector, alpha: float, beta: float, depth: int):
        if depth == 0 or state.is_final():
            if state.turn != self.color:
                return self.value_function(state) * -1, state
            else:
                return self.value_function(state), state
        if state.turn == self.color:
            # maximizing player turn
            max_eval = -np.inf
            next_max_state = state
            for s in StateTransitions.feasible_next_moves(state):
                cur_eval, _ = self.minimax(s, alpha, beta, depth - 1)
                max_eval = max(max_eval, cur_eval)
                if max_eval == cur_eval:
                    next_max_state = s
                alpha = max(alpha, cur_eval)
                if beta <= alpha:
                    break
            return max_eval, next_max_state
        else:
            # minimizing player turn
            min_eval = np.inf
            next_min_state = state
            for s in StateTransitions.feasible_next_moves(state):
                cur_eval, _ = self.minimax(s, alpha, beta, depth - 1)
                min_eval = min(min_eval, cur_eval)
                if min_eval == cur_eval:
                    next_min_state = s
                beta = min(beta, cur_eval)
                if beta <= alpha:
                    break
            return min_eval, next_min_state


class EpsilonAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self,
                 value_function_approximation: BaseJApproximation,
                 color=None,
                 depth=None,
                 epsilon=None,
                 **kwargs):
        super().__init__(value_function_approximation, color, depth)
        self.epsilon = EpsilonGreedyPlayer.DEFAULT_EPSILON if epsilon is None else epsilon

    def next_move(self, state: StateVector, value_func_approx=None) -> StateVector:
        if value_func_approx is None:
            value_func_approx = []
        if not state.is_final():
            if np.random.rand() < 1 - self.epsilon:
                return super().next_move(state, value_func_approx)
            else:
                next_moves = {hash(s): s for s in StateTransitions.feasible_next_moves(state)}
                next_move = rnd.choice(list(next_moves.values()))
                value_func_approx.append(self.value_function.q_factor(state, next_move))
                return next_move
        else:
            return state
