import os
from copy import copy

import numpy as np
import torch

from checkers.function_approximation import BaseJApproximation
from checkers.board import StateVector
from checkers.players import UniformPlayer, CheckersPlayer

from checkers.piece import PieceHelper

from typing import Optional, List, Union, Tuple

from torch.utils.tensorboard import SummaryWriter


class CheckersGym:
    """
    Represents a checkers game.

    Attributes:
    __________
    red_player: CheckersPlayer
        The red player.
    blue_player: CheckersPlayer
        The blue player.
    """
    
    DEFAULT_OUTPUT_PATH = os.path.join('data')
    MAX_NUMBER_OF_MOVES = 100
    
    def __init__(self, light_player: CheckersPlayer, dark_player: CheckersPlayer) -> object:
        self.light_player = light_player
        self.dark_player = dark_player
    
    def __repr__(self):
        return f'{self.light_player}vs{self.dark_player}'
    
    def simulate_game(
            self,
            number_of_moves: Optional[int] = None,
            initial_state: Optional[StateVector] = None
    ) -> Tuple[List[StateVector], int]:
        return self._simulate_game(number_of_moves, initial_state)
    
    def _simulate_game(
            self,
            number_of_moves: Optional[int] = None,
            initial_state: Optional[StateVector] = None,
            value_function: BaseJApproximation = None,
            value_function_history: List = None,
            train: bool = False
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
        if number_of_moves is None:
            number_of_moves = CheckersGym.MAX_NUMBER_OF_MOVES
        
        if initial_state is None:
            initial_state = StateVector()
        
        history = []
        
        if value_function is not None:
            # in this case we assume we need to record every state's value function
            if value_function_history is None:
                raise AttributeError('If the value function wants to be recorded, the value function history list'
                                     ' needs to be passed by reference.')
            
        def record_history(state):
            history.append(state)
        
        current_state = initial_state
        turn = 0
        while True:
            record_history(current_state)
            player = self.dark_player if current_state.turn == PieceHelper.dark else self.light_player
            next_state = player.next_move(current_state, value_func_approx=value_function_history)
            
            # if the game is over (i.e., current player has no moves)
            # or the turn exceeds the maximum number of turns allowed the iteration is stopped.
            if next_state.is_final() or turn > number_of_moves:
                record_history(next_state)
                current_state = copy(next_state)
                break
            else:
                current_state = copy(next_state)
            turn += 1
        
        winner = CheckersGym.decide_winner(current_state)
        
        return history, winner
    
    @staticmethod
    def decide_winner(state):
        """
        Return the winning color.

        Parameters
        ----------
        state: StateVector
            Final state of a game.

        Returns
        -------
        int:
            `PieceHelper.dark` if the dark color wins
            `PieceHelper.light` if the light color wins
            `PieceHelper.empty_square` if the game ends in a tie
        """
        
        min_ = state[:-1].min()
        max_ = state[:-1].max()
        
        if min_ != PieceHelper.empty_square and max_ == PieceHelper.empty_square:
            return PieceHelper.piece_color(min_)
        elif max_ != PieceHelper.empty_square and min_ == PieceHelper.empty_square:
            return PieceHelper.piece_color(max_)
        else:
            return PieceHelper.empty_square
    
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
    
    def td_lambda_training(
            self,
            value_function: BaseJApproximation,
            number_of_games: int,
            gamma: float,
            lambda_: float,
            train: bool = False,
            output_path: str = None,
            writer: torch.utils.tensorboard.SummaryWriter = None
    ):
        if output_path is None:
            output_path = CheckersGym.DEFAULT_OUTPUT_PATH
        
        output_path_training = os.path.join(output_path, 'training_data', str(self))
        output_path_value_function = os.path.join(output_path, 'models', value_function.name)
        os.makedirs(output_path_training, exist_ok=True)
        
        states = []
        targets = []
        
        def update_writer(h):
            if writer:
                writer.add_scalars('Training',
                                   {'Value od Strategy': value_function(h[0])},
                                   value_function.epoch)
                writer.flush()
        
        for i in range(number_of_games):
            value_function_history = []
            history, winner = self._simulate_game(
                value_function=value_function, value_function_history=value_function_history
            )
            
            h1 = history[::2]
            v1 = value_function_history[::2]
            if h1[-1].is_final():
                v1[-1] = float(winner)
            if len(h1) > len(v1):
                v1.append(float(winner))

            h2 = list(map(lambda x: x.flip_colors(), history[1::2]))
            v2 = value_function_history[1::2]
            if h2[-1].is_final():
                v2[-1] = -float(winner)
            if len(h2) > len(v2):
                v2.append(-float(winner))
            
            t1 = self._compute_td_lambda_target(v1, gamma, lambda_, winner)
            t2 = self._compute_td_lambda_target(v2, gamma, lambda_, -winner)
            
            states += h1 + h2
            targets += t1 + t2
            
            if train:
                value_function.train_batch(h1, t1)
                update_writer(h1)
                value_function.train_batch(h2, t2)
                update_writer(h2)
                value_function.save(output_path_value_function, name=f'checkpoint_{value_function.epoch}.pt')
                
        specs = {
            'lambda': lambda_,
            'gamma': gamma,
            'Value function Approximation': str(value_function),
            'Number of games': number_of_games,
            'Light player': self.light_player,
            'Dark player': self.dark_player
        }
        
        CheckersGym.write_training_data(states, targets, output_path_training, specs)
        
    @staticmethod
    def write_training_data(states, targets, output_path, specs):
        with open(os.path.join(output_path, 'specs.txt'), 'w') as f:
            for key, value in specs.items():
                f.write(f'{key}: {value}\n')
            f.close()
        
        targets_ = np.asarray(targets)
        np.savetxt(os.path.join(output_path, 'targets.csv'), targets_, delimiter=",")
        
        states_ = np.concatenate([states])
        np.savetxt(os.path.join(output_path, 'features.csv'), states_, delimiter=",")
    
    @staticmethod
    def _compute_td_lambda_target(value_function_history,
                                  gamma,
                                  lambda_,
                                  winner):
        T = len(value_function_history)
        
        def G(t, n):
            if t + n >= T:
                v = float(winner)
            else:
                v = value_function_history[t + n]
            return (gamma ** n) * v
        
        target = [
            (1 - lambda_) * sum(lambda_ ** n * G(t, n) for n in range(T - t)) + lambda_ ** (T - t) * G(t + 1, T - t - 1)
            for t in range(T)
        ]
        return target
