import numpy as np
from typing import List, Tuple


class PieceHelper:
    """
    Holds the convention for the pieces and some helper functions regarding each piece.
    """
    # Todo: redefine convention. Its computationally cheaper to store ints than floats.
    #  Therefore I suggest using the convention {-2, -1, 0, 1, 2}.
    queen = 2
    piece = 1
    empty_square = 0
    dtype = np.int8
    
    dark = -1
    light = 1
    toggle_turn = dark ^ light
    
    @staticmethod
    @np.vectorize
    def piece_color(piece: int) -> int:
        """
        Return the color of the piece.

        Parameters
        ----------
        piece: int
            The piece value.

        Returns
        -------
            PieceHelper.light or PieceHelper.dark
        """
        if piece == PieceHelper.empty_square:
            return 0
        else:
            return PieceHelper.dark if piece < PieceHelper.empty_square else PieceHelper.light
    
    @staticmethod
    def get_diagonals(piece_value: int) -> List[Tuple[int, int]]:
        """
        Return the possible diagonals in which a piece can move.

        Parameters
        ----------
        piece_value: int
            The value of a piece.

        Returns
        -------
        List[Tuple[int, int]]:
            The list with the possible diagonals in which a piece can go.
        """
        if abs(piece_value) == PieceHelper.queen:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if abs(piece_value) == PieceHelper.piece:
            color = PieceHelper.piece_color(piece_value)
            return [(1 * color, 1), (1 * color, -1)]
