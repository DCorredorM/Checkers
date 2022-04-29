import pandas as pd
import os
import numpy as np

from checkers import StateVector, PieceHelper


def load_training_data(folder):
    states = pd.read_csv(os.path.join(folder, 'features.csv'), dtype=PieceHelper.dtype, header=None)
    targets = pd.read_csv(os.path.join(folder, 'targets.csv'), dtype=np.float32, header=None)
    
    def create_state(row):
        s = StateVector(1) * row.to_numpy()
        s.size_of_the_board = int(np.sqrt((len(s) - 1) * 2))
        return s
    
    states_obj = states.apply(create_state, axis=1).tolist()
    
    return states_obj, targets[0].to_list()
    