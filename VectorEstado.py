import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

class VectorEstado:
    """
        Un VectorEstado recuerda la configuración de piezas en el tablero (un vector de tamaño n^2/2 
        con entradas {−1, −0.5, 0, 0.5, 1} y una componente adicional que recuerda de quién es el turno.

        Negativos: Ocupado por el jugador ROJO.
        Positivos: Ocupado por el jugador AZUL (jugador entrenado).

        0: Casilla vacía.
        0.5: Men.
        1: King.
    """
    def __init__(self, vector = None,turno=(1,None)):
        """
            Se inicializa el tablero de la forma inicial: cada uno de los jugadores en su lado.
        """
        if vector is None:
            self.estado = [0.5,0.5,0.5,0.5,
                            0.5,0.5,0.5,0.5,
                            0.5,0.5,0.5,0.5,
                            0,0,0,0,
                            0,0,0,0,
                            -0.5,-0.5,-0.5,-0.5,
                            -0.5,-0.5,-0.5,-0.5,
                            -0.5,-0.5,-0.5,-0.5]
        else:
            self.estado = vector
        self.turno=turno

    def dar_vector_estado(self):
        """
            Retorna el vector que representa el estado actual del juego.
        """
        return self.estado
    def dar_turno(self):
        """
            Retorna de quien es el turno de jugar.

            El turno está conformado por una tupla.
            La primera parte de la tupla indica el jugador que debe jugar y la segunda, indica la ficha obligada a moverse.
            Si no hay una ficha obligatoria por mover, la segunda entrada es None
           
        """
        return self.turno


    def dar_tablero(self):
        """
            Con la información del vector estado, retorna una matriz con las posiciones de los jugadores y espacios en blanco.
        """
        n = int( np.sqrt(len(self.estado)*2) )
        tablero = np.full((n,n), fill_value=-2.0)

        for i in range(len(self.estado)):
            fila = int( np.floor( 2*i/n ) )
            columna = 2*i%n + fila%2
            tablero[fila,columna] = self.estado[i]
        return tablero

    def visualizar(self, path = None, label = None):
        """
            Se visualiza en un heatmap la configuración del tablero.
        """
        tablero = self.dar_tablero()
        cmapmine = colors.ListedColormap(['white','white', 'red', 'darkred', 'black','blue', 'cornflowerblue'])
        fig, ax = plt.subplots()
        ax.imshow(tablero, cmap=cmapmine, vmin=-2, vmax=1)
        if label is not None:
            plt.legend(label)
        if path is not None:
            plt.savefig(path)
        plt.show()
