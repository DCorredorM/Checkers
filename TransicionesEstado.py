import numpy as np
import VectorEstado

class TransicionesEstado:

    # Convención: Rojos al inicio, negros al final.

    def esvalido(self,  s1: VectorEstado,  s2: VectorEstado):
        """
            Dados dos VectorEstado s1 y s2 decide si pasar de uno al otro en un turno es algo posible 
            según las reglas del juego.
        """
        posibles_jugadas = [x.dar_vector_estado() for x in self.movimientosPosibles(s1)]
        if s2.dar_vector_estado() in posibles_jugadas:
            return True
        else:
            return False

    def movimientosPosiblesPieza(self, s: VectorEstado, p):
        """
            Dado un VectorEstado y una posición p (un entero) ocupada por el jugador en turno lista
            todas las posiciones posibles para esa pieza en un turno según las reglas del juego. 
            Retorna una lista de VectorEstados indicando si la ficha comió o no.
        """ 
        vector = s.dar_vector_estado()
        turno = s.dar_turno()[0]
        pieza = vector[p] 
        tipo_casilla = (p%8 <= 3)                                 # True es tipo a False es tipo b.
        color, tipo_ficha = np.sign(pieza), np.abs(pieza)         # Positivos se mueve a la derecha, negativo se mueve a la izquierda.
                                                                  # 0.5 peón, 1 es reina.  
        if pieza == 0:
            return [] 

        if turno == 1:
            no_comi, comi = 2, 1
        elif turno == 2:
            no_comi, comi = 1, 2

        lista_rta_no_comer = []
        lista_rta_comer = []

        # Posibilidades para peón
        lista_rta_comer, lista_rta_no_comer = self.mueve_adelante_o_atras(vector, tipo_casilla, tipo_ficha, color, pieza, p, comi, no_comi, lista_rta_no_comer, lista_rta_comer)
        # Posibilidades para Reina
        if tipo_ficha==1:
            color_alt=-color
            lista_rta_comer, lista_rta_no_comer = self.mueve_adelante_o_atras(vector, tipo_casilla, tipo_ficha, color_alt, pieza, p, comi, no_comi, lista_rta_no_comer, lista_rta_comer)

        if len(lista_rta_comer) == 0:
            return (lista_rta_no_comer, False)
        else:
            return (lista_rta_comer, True)

    def mueve_adelante_o_atras(self,vector, tipo_casilla, tipo_ficha, color, pieza, p, comi, no_comi, lista_rta_no_comer, lista_rta_comer):
        """
            Retorna una lista de Vector Estado con todos los movimientos posibles de la ficha. 
            La variable color indica hacia donde se mueve la ficha considerada.
            Por ejemplo, como la reina se puede mover a ambos lados, se debe llamar dos veces la función.

            Returns
            -------
            lista_rta_comer : list of Vector Estado
                Lista de estados de llegada posibles después de comer una ficha
            lista_rta_no_comer : list of Vector Estado
                Lista de estados de llegada posibles en los que no se come ficha
        """
        # Posibilidades para peón
        # Se mueve sin comer
        izzz, derrr = 5, 4
        if (not tipo_casilla and  color < 0) or ( tipo_casilla and color > 0):
            izzz -= 1
            derrr -= 1

        iz, der = int(p + color*izzz), int(p + color*derrr)

        if (iz%8 <= 3) != tipo_casilla and (iz in range(len(vector))):
            # Si está vacío me puedo mover
            # Me muevo sin comer
            if vector[iz] == 0:
                nuevo_vector = self.muevo_sin_comer(vector, pieza, color, tipo_ficha, posicion_vieja = p, posicion_nueva = iz)
                lista_rta_no_comer.append(VectorEstado.VectorEstado(nuevo_vector,(no_comi,None)))

            # Si no está vacío, verifico que no sea de mi mismo color
            elif np.sign(vector[iz]) != color:
                muevo_ocho = int(p + color*9)
                if(muevo_ocho in range(len(vector))):
                    if ( muevo_ocho%8 <= 3 ) == tipo_casilla and vector[muevo_ocho] == 0:
                        nuevo_vector = self.muevo_como(vector, pieza, color, tipo_ficha, posicion_vieja = p, posicion_nueva = muevo_ocho, posicion_comi = iz)
                        new_status=(no_comi,None)

                        # Hay que revisar si a partir de esa posición se puede comer (Repetir proceso para el vector nuevo)
                        new_status = self.puede_seguir_comiendo(nuevo_vector, color, posicion_new = muevo_ocho, comi = comi, no_comi = no_comi)
                        lista_rta_comer.append(VectorEstado.VectorEstado(nuevo_vector,new_status))
                            

        if ((der%8 <= 3) != tipo_casilla) and (der in range(len(vector))):
            if vector[der] == 0:
                nuevo_vector = self.muevo_sin_comer(vector, pieza, color, tipo_ficha, posicion_vieja = p, posicion_nueva = der)
                lista_rta_no_comer.append(VectorEstado.VectorEstado(nuevo_vector,(no_comi,None)))
            
            # Si no está vacío, verifico que no sea de mi mismo color
            elif np.sign(vector[der]) != color:
                muevo_ocho = int(p + color*7)
                if(muevo_ocho in range(len(vector))):
                    if ( muevo_ocho%8 <= 3 ) == tipo_casilla and vector[muevo_ocho] == 0:
                        nuevo_vector = self.muevo_como(vector, pieza, color, tipo_ficha, posicion_vieja = p, posicion_nueva = muevo_ocho, posicion_comi = der)
                        new_status=(no_comi,None)
                        
                        # Hay que revisar si a partir de esa posición se puede comer (Repetir proceso para el vector nuevo)
                        new_status = self.puede_seguir_comiendo(nuevo_vector, color, posicion_new = muevo_ocho, comi = comi, no_comi = no_comi)
                        lista_rta_comer.append(VectorEstado.VectorEstado(nuevo_vector,new_status))

        return lista_rta_comer, lista_rta_no_comer


    def muevo_sin_comer(self, vector, pieza, color, tipo_ficha, posicion_vieja, posicion_nueva):
        """
            Retorna el nuevo vector estado después de haber hecho un movimiento simple. 
            Un movimiento simple es aquel en el que la ficha no se come a ninguna otra.

            Returns
            -------
            numpy array
                Retorna el nuevo arreglo de vector estado
        """
        nuevo_vector = vector.copy()
        if tipo_ficha!=1 and ( (posicion_nueva<=31 and posicion_nueva>=28 and color==1) | (posicion_nueva<=3 and posicion_nueva>=0 and color==-1)):
            nuevo_vector[posicion_vieja], nuevo_vector[posicion_nueva] = 0, color
        else:
            nuevo_vector[posicion_vieja], nuevo_vector[posicion_nueva] = 0, pieza
        return nuevo_vector

    def muevo_como(self, vector, pieza, color, tipo_ficha, posicion_vieja, posicion_nueva, posicion_comi):
        """
            Retorna el nuevo vector estado después de comer una ficha. 
            Debe actualizar la casilla de posición inicial, la que se comió y la posición donde quedí.

            Returns
            -------
            numpy array
                Retorna el nuevo arreglo de vector estado
        """
        nuevo_vector = vector.copy()
        if tipo_ficha!=1 and ( (posicion_nueva<=31 and posicion_nueva>=28 and color==1) | (posicion_nueva<=3 and posicion_nueva>=0 and color==-1)):
            nuevo_vector[posicion_vieja], nuevo_vector[posicion_comi], nuevo_vector[posicion_nueva] = 0, 0, color
        else:
            nuevo_vector[posicion_vieja], nuevo_vector[posicion_comi], nuevo_vector[posicion_nueva] = 0, 0, pieza
        return nuevo_vector

    def puede_seguir_comiendo(self, nuevo_vector, color, posicion_new, comi, no_comi):
        """
            Determina si una ficha que ya comió puede seguir comiendo. 
            Retorna una tupla new_status = (debo_comer, posicion_ficha)
            Donde debo_comer indica si en el siguiente turno la ficha debe seguir comiento o no.
            Además, si debo_comer es True, entonces se debe guardar la ficha con la que se debe comer en posicion_ficha.

            Returns 
            -------
            new_status : (int, int)
                La primera entrada determina si se debe comer en la siguiente jugada.
                Si es True, entonces indica con qué ficha debe comer.
        """
        new_status = (no_comi, None)
        tipo_casilla_new = (posicion_new%8 <= 3)
        izzz_new, derrr_new = 5, 4
        if (not tipo_casilla_new and  color < 0) or ( tipo_casilla_new and color > 0):
            izzz_new -= 1
            derrr_new -= 1
        iz_new, der_new = int(posicion_new + color*izzz_new), int(posicion_new + color*derrr_new)
        
        if ((iz_new%8 <= 3) != tipo_casilla_new) and (iz_new in range(len(nuevo_vector))):
            # Si la casilla está vacia, ie vector == 0 no queremos verificar nada 
            if (np.sign(nuevo_vector[iz_new]) != color) and (nuevo_vector[iz_new] !=0):
                muevo_ocho_new = int(posicion_new + color*9)
                if(muevo_ocho_new in range(len(nuevo_vector))):
                    if ( muevo_ocho_new%8 <= 3 ) == tipo_casilla_new and nuevo_vector[muevo_ocho_new] == 0:
                        new_status=(comi, posicion_new)    
        if ((der_new%8 <= 3) != tipo_casilla_new ) and (der_new in range(len(nuevo_vector))):
            # Si la casilla está vacia, ie vector == 0 no queremos verificar nada 
            if (np.sign(nuevo_vector[der_new]) != color) and (nuevo_vector[der_new]!=0):
                muevo_ocho_new = int(posicion_new + color*7)
                if(muevo_ocho_new in range(len(nuevo_vector))):
                    if ( muevo_ocho_new%8 <= 3 ) == tipo_casilla_new and nuevo_vector[muevo_ocho_new] == 0:
                        new_status=(comi,posicion_new)      
        return new_status

    def movimientosPosibles(self, s: VectorEstado):
        """
            Dado un VectorEstado s retorna la lista de VectorEstados de todas las jugadas admisibles en
            un turno iniciando en s.
        """
        movimientos_posibles_no_come = []
        movimientos_posibles_come = []

        turno = pieza_mover = s.dar_turno()[0]
        pieza_mover = s.dar_turno()[1]
        vector = s.dar_vector_estado()

        if pieza_mover is not None:
            mov_posibles_ficha, comio = self.movimientosPosiblesPieza(s,pieza_mover)
            if comio:
                movimientos_posibles_come = mov_posibles_ficha

        else:
            for x in range(len(vector)):
                if (turno == 1 and vector[x] > 0) or (turno == 2 and vector[x] < 0):
                    mov_posibles_ficha, comio = self.movimientosPosiblesPieza(s,int(x))
                    if comio:
                        for t in self.movimientosPosiblesPieza(s,int(x))[0]:
                            movimientos_posibles_come.append(t)
                    else:
                        for t in self.movimientosPosiblesPieza(s,int(x))[0]:
                            movimientos_posibles_no_come.append(t)
        
        if len(movimientos_posibles_come) == 0:
            return movimientos_posibles_no_come
        else:
            return movimientos_posibles_come
