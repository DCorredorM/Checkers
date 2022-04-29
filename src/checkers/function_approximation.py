import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from checkers.board import StateVector, StateTransitions

import torch
from torch import nn


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
    def __init__(self, name=None, **kwargs):
        self.name = self.__class__.__name__ if name is None else name
        self.epoch = kwargs.pop('epoch', 0)
    
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
    def save(self, path, **kwargs):
        """
        Saves an already trained model
        
        Returns
        -------

        """
        ...
    
    @classmethod
    @abstractmethod
    def load(cls, path, **kwargs):
        ...
    
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
    
    def save(self, path, **kwargs):
        pass

    @classmethod
    def load(cls, path, **kwargs):
        pass

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


class NNetApprox(nn.Module, BaseJApproximation):
    
    def __init__(self, size_of_board=8, **kwargs):
        super().__init__()
        super(nn.Module, self).__init__(**kwargs)
        
        net = kwargs.pop('net', None)
        self.size_of_board = size_of_board
        if net is None:
            net = self.build_net(**kwargs)
        self.net: nn.Sequential = net

        loss = kwargs.pop('loss', None)
        if loss is None:
            loss = nn.MSELoss()
        self.loss = loss
        
        optimizer = kwargs.pop('optimizer', None)
        if optimizer is None:
            lr = kwargs.pop('optimizer', 0.01)
            momentum = kwargs.pop('optimizer', 0.9)
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        
        self.optimizer = optimizer
    
    def __call__(self, states: Union[StateVector, List[StateVector]], tensor=False, **kwargs):
        if tensor:
            return self.forward(states, **kwargs)
        else:
            return self._evaluate_function(states, **kwargs)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.epoch})'

    @staticmethod
    def build_net(structure, size_of_board=8, **kwargs):
        structure_ = [nn.Linear((size_of_board ** 2) // 2 + 1, structure[0]), nn.ReLU()]
        structure_ += sum(
            [[nn.Linear(structure[i], structure[i + 1]), nn.ReLU()] for i in range(len(structure) - 1)],
            []
        )
        structure_ += [nn.Linear(structure[-1], 1)]
        net = nn.Sequential(*structure_)
        return net
    
    def save(self, path, name='checkpoint.pt', **kwargs):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'net': self.net,
            'size_of_board': self.size_of_board,
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer': self.optimizer,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'name': self.name
        }, os.path.join(path, name))

    @classmethod
    def load(cls, path, name='checkpoint.pt', train=True, **kwargs):
        checkpoint = torch.load(os.path.join(path, name))
        
        obj = cls(**checkpoint)
        obj.net.load_state_dict(checkpoint['model_state_dict'])
        obj.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if train:
            obj.train()
        else:
            obj.eval()
        return obj
        
    @staticmethod
    def state_to_tensor(states: Union[StateVector, List[StateVector]]) -> torch.Tensor:
    
        if isinstance(states, tuple):
            states = states[0]
        if isinstance(states, list):
            states = np.vstack(states)
        return torch.Tensor(states)

    def train_step(self, state: StateVector, target: float) -> None:
        pass

    def train_batch(self, states: List[StateVector], target: List[float]) -> None:
        self.epoch += 1
        running_loss = 0.
        last_loss = 0.
        
        inputs = NNetApprox.state_to_tensor(states)
        targets = torch.Tensor([target]).T
            
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()

        # Make predictions for this batch
        outputs = self.forward(inputs)

        # Compute the loss and its gradients
        loss = self.loss(outputs, targets)
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()

        # Gather data and report
        running_loss += loss.item()
            
        return loss.item()
    
    def forward(self, *input, **kwargs) -> torch.Tensor:
        x = NNetApprox.state_to_tensor(input)
        value = self.net(x)
        return value
        
    def _evaluate_function(self, state: StateVector, as_list=False) -> Union[float, np.ndarray, List[float]]:
        result = self.forward(state)
        
        result = np.array(result.tolist())
        if len(result) == 1:
            return result[0]
        else:
            return result
    
    
    
    
