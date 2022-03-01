import TransicionesEstado as te
import VectorEstado
import JugadorUniforme as jugadorU

class JuegoDAMAS:
    def __init__(self):
        self.jugador1=1
        self.jugador1_color='AZUL'
        self.jugador2=2
        self.jugador2_color='ROJO'

    def simula(self, NumJugadas, vector: VectorEstado, guardar = False):
        jugador = jugadorU.JugadorUniforme()
        vector.visualizar()
        jugada=0
        while jugada < NumJugadas:
            vector = jugador.nextMove(vector)
            if vector is None: 
                print('Ya no hay más movimientos posibles.')
                break
            print(f'Vector estado turno {vector.dar_turno()}')
            print(f'Se visualiza la jugada número {jugada+1}')
            if guardar:
                vector.visualizar(path = 'estado' + '-' + str(jugada) + '.png', label = 'Jugada número ' + str(jugada))
            else:
                vector.visualizar()
            jugada+=1

    