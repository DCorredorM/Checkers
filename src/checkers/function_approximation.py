from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from checkers.board import StateVector, StateTransitions


class BaseQApproximation(ABC):
    def __init__(self):
        pass

    def __call__(
            self, state_action: Union[List[Tuple[StateVector, StateVector]], Tuple[StateVector, StateVector]]
    ):
        if isinstance(state_action, list):
            return list(map(lambda x: self._evaluate_function(*x), state_action))
        else:
            return self._evaluate_function(*state_action)
    
    @abstractmethod
    def train_step(
            self, state: StateVector, action: StateVector, target: float
    ) -> None:
        """
        Performs a training step for one single observation.
        
        Parameters
        ----------
        state: StateVector
        action: StateVector
        target: float
        """
        ...

    @abstractmethod
    def train_batch(
            self, state_action: List[Tuple[StateVector, StateVector]], target: List[float]
    ) -> None:
        ...
    
    @abstractmethod
    def _evaluate_function(
            self, state: StateVector, action: StateVector
    ) -> float:
        """
        Evaluate the approximation in the given state action pair(s).
        
        Parameters
        ----------
        state: StateVector
        action: StateVector
        
        Returns
        -------
        float
            The approximation of the Q factors.
        """
        ...


class BaseJApproximation(ABC):
    def __init__(self):
        pass
    
    def __repr__(self):
        return self.__class__.__name__
    
    def __call__(
            self, state: Union[List[StateVector], StateVector]
    ):
        if isinstance(state, list):
            return list(map(lambda x: self._evaluate_function(*x), state))
        else:
            return self._evaluate_function(state)
    
    @abstractmethod
    def save(self):
        """
        Saves an already trained model
        
        Returns
        -------

        """
        ...
    
    @classmethod
    @abstractmethod
    def load(cls):
    
    @abstractmethod
    def train_step(
            self, state: StateVector, target: float
    ) -> None:
        """
        Performs a training step for one single observation.

        Parameters
        ----------
        state: StateVector
        action: StateVector
        target: float
        """
        ...
    
    @abstractmethod
    def train_batch(
            self, states: List[StateVector], target: List[float]
    ) -> None:
        ...
    
    @abstractmethod
    def _evaluate_function(
            self, state: StateVector
    ) -> float:
        """
        Evaluate the approximation in the given state action pair(s).

        Parameters
        ----------
        state: StateVector

        Returns
        -------
        float
            The approximation of the Q factors.
        """
        ...
    
    def q_factor(self, state: StateVector, action: StateVector):
        if action.is_final():
            return self(action)
        else:
            return min(self(s) for s in StateTransitions.feasible_next_moves(action))
        

class MaterialBalanceApprox(BaseJApproximation):
    
    def train_step(self, state: StateVector, target: float) -> None:
        pass

    def train_batch(self, states: List[StateVector], target: List[float]) -> None:
        pass

    def _evaluate_function(self, state: StateVector) -> float:
        my_pieces = sum(state[i] for i in state.get_pieces_in_turn()) * state.turn
        state.toggle_turn()
        opponents_pieces = sum(state[i] for i in state.get_pieces_in_turn()) * state.turn
        state.toggle_turn()
    
        return 2 * my_pieces / (my_pieces + opponents_pieces) - 1
