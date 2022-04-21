import numpy as np
from typing import List, Tuple, Union
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

from gui.mpl_visualizer import Visualizer
from checkers.piece import PieceHelper
from utils import flatten_list


class StateVector(np.ndarray):

    def __new__(cls, size_of_the_board: int = 8, **kwargs):
        """
        The constructor of the class
        """

        # The size of the vector.
        n = size_of_the_board ** 2 // 2 + 1

        kwargs['shape'] = (n,)
        kwargs['dtype'] = PieceHelper.dtype

        obj = super().__new__(cls, **kwargs)

        obj.__init__(size_of_the_board)
        return obj

    # noinspection PyMissingConstructor
    def __init__(self, size_of_the_board: int = 8):
        self.size_of_the_board = size_of_the_board

        pieces_per_row = self.size_of_the_board // 2

        # set the value for the light pieces
        self[0: pieces_per_row * 3] = PieceHelper.piece
        self[pieces_per_row * 3: -pieces_per_row * 3 - 1] = PieceHelper.empty_square
        self[-(pieces_per_row * 3 + 1):-1] = -PieceHelper.piece

        self.turn = PieceHelper.light

    def __array_finalize__(self, obj):

        if obj is None: return

        self.size_of_the_board = getattr(obj, 'size_of_the_board', 8)
    
    def __hash__(self):
        return hash(str(self))
    
    @property
    def turn(self):
        return self[-1]

    @turn.setter
    def turn(self, value):
        self[-1] = value

    def toggle_turn(self):
        self.turn ^= PieceHelper._toggle_turn

    def visualize(self, show=True, save_path=None) -> None:
        """Visualize the state."""
        Visualizer.visualize_state(self)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        if save_path is not None:
            plt.clf()

    def get_pieces_in_turn(self) -> List[int]:
        """
        Return all the positions in which the player in turn has pieces.

        Returns
        -------
            Return all the positions in which the player of color has pieces.
        """
        return list(*np.where(PieceHelper.piece_color(self[:-1]) == self.turn))

    def from_index_to_coordinate(self, index_: int) -> Tuple[int, int]:
        """
        Maps the index of the array into a coordinate representing it's location on the board.

        Parameters
        ----------
        index_: int
            index to be mapped.

        Returns
        -------
        Tuple[int, int]:
            Coordinate representing the index location on the board.

        """
        if index_ > len(self) - 1:
            raise Exception('The given index does not lies within the board. '
                            f'The index must be less than {len(self) - 1}')
        k = index_ * 2
        n = self.size_of_the_board

        i = k // n
        j = k % n + 1 - i % 2
        return i, j

    def from_coordinate_to_index(self, coordinate: Tuple[int, int]) -> int:
        """
        Map a coordinate into the index that represents it.

        Parameters
        ----------
        coordinate: Tuple[int, int]
            coordinate to be mapped.

        Returns
        -------
        Tuple[int, int]:
            Index representing the coordinate.

        """
        n = self.size_of_the_board
        i, j = coordinate

        return (i * n + j) // 2

    def coordinate_in_board(self, coordinate: Tuple[int, int]) -> bool:
        """
        Return true if the coordinate is in the board, false otherwise.

        Parameters
        ----------
        coordinate: Tuple[int, int]
            Coordinate to evaluate

        Returns
        -------
        bool:
            True if the coordinate is in the board, false otherwise.
        """
        i, j = coordinate
        return 0 <= i < self.size_of_the_board and 0 <= j < self.size_of_the_board

    def check_promotion(self, piece_index, piece_value):
        """
        Check if the given piece should be promoted.

        Parameters
        ----------
        piece_index
        piece_value

        Returns
        -------


        """
        if abs(piece_value) == PieceHelper.queen:
            return False
        else:
            i, j = self.from_index_to_coordinate(piece_index)

            if PieceHelper.piece_color(piece_value) == PieceHelper.light:
                return True if i == self.size_of_the_board - 1 else False
            else:
                return True if i == 0 else False
    
    def is_final(self):
        s_list = sorted(list(self[:-1]))
        if s_list[0] == 0 or s_list[-1] == 0:
            # In this case all pieces are of the same color
            return True
        else:
            return False
        
    
# Todo: Should we merge these two classes together??
class StateTransitions:
    def __init__(self):
        pass

    @staticmethod
    def is_valid(source_state: StateVector, target_state: StateVector) -> bool:
        """
        Check if the transition between source_state and target_state is valid.

        Parameters
        ----------
        source_state: StateVector
            Source state.
        target_state: StateVector
            Tentative next state.

        Returns
        -------
        bool:
            True if it is possible to transition from source_state into target_state given the game roules.
        """
        pass

    @staticmethod
    def feasible_moves_piece(state: StateVector, piece_index: int) -> List[StateVector]:
        """
        Return the feasible moves for a piece in a given state.

        Given a StateVector and a position, it returns all the feasible moves for that piece, i.e., the feasible
        StateVector in which the given piece is moved.

        Parameters
        ----------
        state: StateVector:
            The state in which the move will be made.
        piece_index: int
            An integer or tuple representing the piece for which the next states will be computed.

        Returns
        -------
        List[StateVector]:
            List of StateVector with the feasible states resulting from moving the given piece.
        """
        i, j = state.from_index_to_coordinate(piece_index)
        states = flatten_list([
            StateTransitions._feasible_state_diagonal(state, piece_index, d, from_jump=False)
            for d in PieceHelper.get_diagonals(piece_value=state[piece_index])
            if state.coordinate_in_board((i + d[0], j + d[1]))
        ])

        return StateTransitions._filter_only_take(state, states)

    @staticmethod
    def _feasible_state_diagonal(
            state: StateVector,
            piece_index: int,
            diagonal: Tuple[int, int],
            from_jump: bool = False) -> List[StateVector]:
        if PieceHelper.piece_color(state[piece_index]) != state.turn:
            logging.warning('You are trying to move a piece in a wrong turn. '
                            'Either moving a dark piece in lights turn or light piece in dark turn.')
            return []

        i, j = state.from_index_to_coordinate(piece_index)
        d_i, d_j = diagonal

        diag_index = state.from_coordinate_to_index((i + d_i, j + d_j))
        diag2_index = state.from_coordinate_to_index((i + 2 * d_i, j + 2 * d_j))
        if state.coordinate_in_board(coordinate=(i + 2 * d_i, j + 2 * d_j)):
            # boolean variable that indicates if the current piece can jump.
            # A piece can jump in a given diagonal if the immediate next square is of
            # the other color and the next square is available.
            _can_jump = (PieceHelper.piece_color(state[piece_index]) != PieceHelper.piece_color(state[diag_index])
                         and state[diag2_index] == PieceHelper.empty_square
                         and state[diag_index] != PieceHelper.empty_square)
        else:
            _can_jump = False

        if _can_jump:
            # If the piece can jump, then jump, eat next piece and evaluate that state to see if it can keep jumping.
            # i.e., check if you can keep jumping or not.
            new_state = deepcopy(state)
            piece_value = state[piece_index]

            # remove piece from initial square
            new_state[piece_index] -= piece_value

            # Take middle piece, i.e., empty square.
            new_state[diag_index] = PieceHelper.empty_square

            promote = new_state.check_promotion(diag2_index, piece_value)

            if promote:
                # add piece to target square and promote
                new_state[diag2_index] += PieceHelper.piece_color(piece_value) * PieceHelper.queen
            else:
                # add piece to target square
                new_state[diag2_index] += piece_value

            i, j = new_state.from_index_to_coordinate(diag2_index)

            states = flatten_list(
                [StateTransitions._feasible_state_diagonal(new_state, diag2_index, d, from_jump=True)
                 for d in PieceHelper.get_diagonals(piece_value=new_state[diag2_index])
                 if state.coordinate_in_board((i + d[0], j + d[1]))])
            return states

        elif state[diag_index] == PieceHelper.empty_square and from_jump is False:
            # diagonal square is empty and the piece does not come from a jump
            # Your only choice in this diagonal is to move one square in that direction.
            new_state = deepcopy(state)
            piece_value = state[piece_index]
            new_state[piece_index] -= piece_value

            promote = new_state.check_promotion(diag_index, piece_value)

            if promote:
                new_state[diag_index] += PieceHelper.piece_color(piece_value) * PieceHelper.queen
            else:
                new_state[diag_index] += piece_value

            # Move to diagonal and change turn
            new_state.toggle_turn()

            return [new_state]
        elif from_jump is True: #state[diag_index] == PieceHelper.empty_square and
            # If the piece can't neither jump nor move to the adjacent diagonal square then it cannot do anything.

            new_state = deepcopy(state)
            # Change turn
            new_state.toggle_turn()

            return [new_state]
        else:
            return []

    @staticmethod
    def feasible_next_moves(state: StateVector) -> List[StateVector]:
        """
        Return all the possible next moves.

        Parameters
        ----------
        state: StateVector:
            The state in which the move will be made.

        Returns
        -------
        List[StateVector]:
            List of StateVector with the feasible states that the game can reach in one step.
        """
        pieces_to_move = state.get_pieces_in_turn()
        states = flatten_list([StateTransitions.feasible_moves_piece(state, piece) for piece in pieces_to_move])
        feasible_moves = StateTransitions._filter_only_take(state, states)
        return feasible_moves

    @staticmethod
    def _filter_only_take(state, states):
        piece_imbalance = int(abs(state[:-1].sum()))
        if states:
            # force taking opponets pieces if possible
            imbalance_deltas = [abs(piece_imbalance - int(abs(s[:-1].sum()))) for s in states]
            max_pieces_imbalance_delta = max(imbalance_deltas)
            stop = 0
            p_len = len(states)
            if min(imbalance_deltas) != max(imbalance_deltas):
                stop += 1

            states = list(
                filter(lambda s: abs(piece_imbalance - int(abs(s[:-1].sum()))) >= max_pieces_imbalance_delta, states)
            )
            if stop > 0 and p_len == len(states):
                print('stop')

        return states



