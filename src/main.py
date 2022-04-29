import os

import torch
from torch import nn

from checkers import NNetApprox, MaterialBalanceApprox, CheckersGym, AlphaBetaPlayer, EpsilonAlphaBetaPlayer, \
    UniformPlayer

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

from utils.data_handler import load_training_data


def build_net():
    path = os.path.join('data', 'models', 'power_net')
    structure = [512] * 3 + [256] * 3

    net = NNetApprox.build_net(structure=structure)
    
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )
    
    loss = nn.MSELoss()
    
    nnet = NNetApprox(net=net, loss=loss, optimizer=optimizer)

    dataset = 'UniformPlayer(-1)vsAlphaBetaPlayer(1)(4)'
    states, targets = load_training_data(
        os.path.join('data', 'training_data', dataset)
    )
    
    for e in range(100):
        loss = nnet.train_batch(states, targets)
        print(loss)
    
    nnet.epoch = 0
    nnet.save(path, name='initial.pt')


def choose_last_checkpoint(path):
    dir = os.listdir(path)
    
    def order(name: str):
        name = name.strip('.pt')
        if 'checkpoint' in name:
            return int(name.replace('checkpoint', ''))
        else:
            return -1
    
    return max(dir, key=lambda x: order(x))


def train(number_of_games=300,
          gamma=0.9,
          lambda_=0.8):
    
    writer = SummaryWriter('runs/power_net')
    
    net_path = os.path.join('data', 'models', 'power_net')
    name = choose_last_checkpoint(net_path)
    
    light_approx = NNetApprox.load(net_path, name=name)
    light_approx.optimizer.param_groups[0]['lr'] = 0.01
    
    v_approx = MaterialBalanceApprox()
    light_player = EpsilonAlphaBetaPlayer(v_approx, depth=6, epsilon=0.1)
    
    dark_player = EpsilonAlphaBetaPlayer(v_approx, depth=4, epsilon=0.1)
    # dark_player = UniformPlayer(v_approx)
    
    game = CheckersGym(
        light_player=light_player,
        dark_player=dark_player
    )
    
    game.td_lambda_training(
        value_function=light_approx,
        number_of_games=number_of_games,
        gamma=gamma,
        lambda_=lambda_,
        train=True,
        writer=writer
    )
    light_approx.save(net_path, name=f'checkpoint{light_approx.epoch}.pt')


if __name__ == '__main__':
    build_net()
    train()