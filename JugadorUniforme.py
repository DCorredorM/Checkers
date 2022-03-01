import random

from sympy import EX
import VectorEstado
import TransicionesEstado

class JugadorUniforme:

    def nextMove(self,s):
        """
            Elige su jugada de manera aleatoria uniforme entre las jugadas válidas disponibles.
        """
        te = TransicionesEstado.TransicionesEstado()
        posibles_jugadas = te.movimientosPosibles(s)
        if len(posibles_jugadas) == 0:
            return None
        print(f'Tamaño de posibles_jugadas {len(posibles_jugadas)}')
        return random.choice(posibles_jugadas)

