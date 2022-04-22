from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import glob

import numpy as np
import os
from typing import List

from checkers.piece import PieceHelper


class Visualizer:
    piece_images = {PieceHelper.dark * PieceHelper.queen: os.path.realpath('data/gui_data/black_queen.png'),
                    PieceHelper.dark * PieceHelper.piece: os.path.realpath('data/gui_data/black_piece.png'),
                    PieceHelper.light * PieceHelper.queen: os.path.realpath('data/gui_data/white_queen.png'),
                    PieceHelper.light * PieceHelper.piece: os.path.realpath('data/gui_data/white_piece.png')}

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def plain_board(dimension=8):
        n = dimension
        image = np.array([[1 if (i + j) % 2 == 0 else 0 for j in range(n)] for i in range(n)], dtype=float)
        return image

    @staticmethod
    def board(dimension=8, fig=None, ax=None):
        n = dimension
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        else:
            pass
            # ax.clear()
        board = Visualizer.plain_board(n)
        row_labels = list(range(1, n + 1))
        col_labels = list(range(1, n + 1))
        ax.matshow(board, cmap='gist_ncar')
        plt.xticks(range(n), col_labels)
        plt.yticks(range(n), row_labels)

        return fig, ax

    @staticmethod
    def plot_piece(ax, i, j, piece: int):
        arr_lena = mpimg.imread(Visualizer.piece_images[piece])

        imagebox = OffsetImage(arr_lena, zoom=0.3)
        ab = AnnotationBbox(imagebox, (j, i), frameon=False)
        ax.add_artist(ab)
        plt.draw()

    @staticmethod
    def visualize_state(board: 'StateVector', fig=None, ax=None):
        fig, ax = Visualizer.board(board.size_of_the_board, fig, ax)

        for k in range(len(board) - 1):
            piece = board[k]
            if piece != PieceHelper.empty_square:
                i, j = board.from_index_to_coordinate(k)
                Visualizer.plot_piece(ax, i, j, board[k])

        return fig, ax

    @staticmethod
    def visualize_game(states: List['StateVector'], output_path, **kwargs):
        path = os.path.join(output_path, 'frames')
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(states):
            s.visualize(save_path=os.path.join(path, f'frame{i}.jpg'), show=False)
        
        name = 'game'
        Visualizer.create_video_from_folder(path, name, **kwargs)
        
    @staticmethod
    def create_video_from_folder(image_folder, video_name, **kwargs):
        cur_path = os.path.realpath(os.curdir)
        os.chdir(image_folder)
        
        framerate = kwargs.get('framerate', 5)
        fps = kwargs.get('fps', 25)
        vcodec = kwargs.get('vcodec', 'mpeg4')
        vb = kwargs.get('vb', '40M')
        
        os.popen(
            f'ffmpeg -framerate {framerate} -i frame%01d.jpg  -vf fps={fps} -vcodec {vcodec} -y -vb {vb} {video_name}.mp4'
        ).readlines()
        os.rename(f'{video_name}.mp4', f'../{video_name}.mp4')
        os.chdir(cur_path)
        