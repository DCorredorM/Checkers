import os.path
from abc import abstractmethod
import time
from typing import List

import pandas as pd
import plotly.express as px

import checkers
from ipywidgets import widgets
from ipywidgets import Video

from IPython.display import Image

import gui

PLAYERS = {
    'UniformPlayer': checkers.players.UniformPlayer,
    'EpsilonGreedyPlayer': checkers.players.EpsilonGreedyPlayer,
    'AlphaBetaPlayer': checkers.players.AlphaBetaPlayer,
    'EpsilonAlphaBetaPlayer': checkers.players.EpsilonAlphaBetaPlayer
}

VALUE_FUNCTIONS = {
    'MaterialBalanceApprox': checkers.function_approximation.MaterialBalanceApprox,
    'NNetApprox': checkers.function_approximation.NNetApprox
}

NET_PATH = os.path.join('.')
CACHE = os.path.join('data', '.AppCache')


class _SectionBase(object):
    OUT_LAYOUT = widgets.Layout(
        # height='1000px', width='1000px', margin='0 50px 0 50px', padding='0 20px 0 20px',
        # justify_content='center',
        # justify_items='center',
        # align_content='center'
    )
    
    def __init__(self, *args, **kwargs):
        self.widgets = {}
        self.out = widgets.Output()
        
        self.build_widgets()
        self.link_widgets()
    
    @abstractmethod
    def build_widgets(self):
        ...
    
    def link_widgets(self):
        for _, w in self.widgets.items():
            w.observe(self.on_change)
    
    def display(self):
        out = widgets.VBox([
            widgets.HBox(list(self.widgets.values())),
            self.out
        ], layout=_SectionBase.OUT_LAYOUT)
        
        self.update_output()
        return out
    
    @abstractmethod
    def update_output(self):
        ...
    
    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.update_output()


class PlayerController(_SectionBase):
    
    def __init__(self):
        super(PlayerController, self).__init__()
    
    def build_widgets(self):
        style = {'description_width': 'initial'}
        players_ = list(PLAYERS.keys())
        self.widgets['player'] = widgets.Dropdown(
            options=players_,
            value=players_[0],
            description=f'Choose player:',
            disabled=False,
            style=style
        )
        v_funcs = list(VALUE_FUNCTIONS.keys())
        self.widgets['value_function'] = widgets.Dropdown(
            options=v_funcs,
            value=v_funcs[0],
            description=f'Value Function:',
            disabled=False,
            style=style
        )
        # only when player is alpha beta
        self.widgets['depth'] = widgets.IntSlider(
            value=4,
            min=0,
            max=10,
            step=1,
            description='Depth:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style=style
        )
        # only when function is NNet
        possible_nets = self.load_possible_nets()
        self.widgets['net_file'] = widgets.Dropdown(
            options=possible_nets,
            value=possible_nets[0],
            description=f'Choose Net:',
            disabled=False,
            style=style
        )
        
        self.widgets['epsilon'] = widgets.FloatSlider(
            value=0.1,
            min=0,
            max=1,
            step=0.01,
            description='Epsilon:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            style=style
        )
    
    def _hide_all(self):
        for w in self.widgets.values():
            w.layout.visibility = 'hidden'
    
    def update_output(self):
        self._hide_all()
        self.widgets['player'].layout.visibility = 'visible'
        
        if self.widgets['player'].value in ['AlphaBetaPlayer', 'EpsilonAlphaBetaPlayer', 'EpsilonGreedyPlayer']:
            self.widgets['value_function'].layout.visibility = 'visible'
            
            if self.widgets['player'].value in ['AlphaBetaPlayer', 'EpsilonAlphaBetaPlayer']:
                self.widgets['depth'].layout.visibility = 'visible'
            
            if self.widgets['player'].value in ['EpsilonAlphaBetaPlayer', 'EpsilonGreedyPlayer']:
                self.widgets['epsilon'].layout.visibility = 'visible'
            
            if self.widgets['value_function'].value in ['NNetApprox']:
                self.widgets['net_file'].layout.visibility = 'visible'
    
    @staticmethod
    def load_possible_nets(path=None) -> List:
        return ['']
    
    def observe(self, *args, **kwargs):
        for w in self.widgets.values():
            w.observe(*args, **kwargs)
    
    def create_player(self):
        if self.widgets['value_function'].value == 'NNetApprox':
            path = os.path.join(NET_PATH, self.widgets['net_file'].value)
            vf = VALUE_FUNCTIONS['NNetApprox'].load(path=path)
        else:
            vf = VALUE_FUNCTIONS[self.widgets['value_function'].value]()
        kwargs = dict(
            value_function_approximation=vf,
            depth=self.widgets['depth'].value,
            epsilon=self.widgets['epsilon'].value
        )
        return PLAYERS[self.widgets['player'].value](**kwargs)


class PlayerManager(_SectionBase):
    def __init__(self):
        super().__init__()
        self.changed = True
    
    def build_widgets(self):
        self.widgets['light'] = PlayerController()
        self.widgets['dark'] = PlayerController()
    
    def update_output(self):
        self.widgets['light'].update_output()
        self.widgets['dark'].update_output()
        self.changed = True
    
    def display(self):
        players = widgets.Tab([
            widgets.VBox(list(self.widgets['light'].widgets.values())),
            widgets.VBox(list(self.widgets['dark'].widgets.values()))
        ])
        players.set_title(0, 'Light player')
        players.set_title(1, 'Dark player')
        
        out = widgets.VBox([
            players,
            self.out
        ], layout=_SectionBase.OUT_LAYOUT)
        
        self.update_output()
        return out


class MultiGame(_SectionBase):
    def __init__(self, player_manager, **kwargs):
        self.player_manager = player_manager
        self.game = self.create_game()
        self.history = {str(self.game): []}
        
        self.out1 = widgets.Output()
        self.out2 = widgets.Output()
        
        self.image_index = 0
        super().__init__(**kwargs)
    
    def create_game(self):
        self.player_manager.changed = False
        return checkers.CheckersGym(
            light_player=self.player_manager.widgets['light'].create_player(),
            dark_player=self.player_manager.widgets['dark'].create_player())
    
    def build_widgets(self):
        # first tab
        self.widgets['play_game'] = widgets.Button(
            description='Simulate',
            disabled=False,
            # button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click To Simulate',
            icon='atom'  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.widgets['play_game'].on_click(self.on_simulate)
        
        self.widgets['plot'] = widgets.ToggleButton(
            value=True,
            description='Render',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Toggle to save rendered game',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        
        # second tab
        possible_rendered = MultiGame.ready_to_render_files()
        self.widgets['game_video'] = widgets.Dropdown(
            options=possible_rendered,
            value=possible_rendered[0],
            description='Game:',
            disabled=False,
        )
        self.widgets['game_video'].observe(self.rese_image_index)

        self.widgets['video'] = widgets.ToggleButton(
            value=True,
            description='Video',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Toggle to play video or show frames',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        self.widgets['next'] = widgets.Button(
            description='Next',
            disabled=False,
            # button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='next image',
            icon='angle-right'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['next'].on_click(self.on_next)

        self.widgets['prev'] = widgets.Button(
            description='Previous',
            disabled=False,
            # button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Previous image',
            icon='angle-left'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['prev'].on_click(self.on_prev)
        
        # third tab

        possible_rendered = MultiGame.ready_to_summary_files()
        self.widgets['summary'] = widgets.Dropdown(
            options=possible_rendered,
            value=possible_rendered[0],
            description='Game:',
            disabled=False,
        )
    
    def on_simulate(self, b):
        self.update_game()
        message = {1: "Winner was the light player!", 0: "Game ended in tie...", -1: "Winner was the dark player!"}
        with self.out:
            self.out.clear_output(True)
            print(f'Playing {self.game}')
            history, winner = self.play()
            print(message[int(winner)])
            if self.widgets['plot'].value:
                print(f'Preparing render...')
                self.save(history)
                print(f'Ready to render!')
                
    def on_next(self, b):
        name = self.widgets['game_video'].value
        if name is not None:
            if os.path.exists(os.path.join(CACHE, 'games', name, 'frames', f'frame{self.image_index + 1}.jpg')):
                self.image_index += 1
    
            self.update_output2()

    def rese_image_index(self, *args):
        self.image_index = 0
    
    def on_prev(self, b):
        name = self.widgets['game_video'].value
        if name is not None:
            if self.image_index > 0:
                self.image_index -= 1
            
            self.update_output2()
    
    def update_output1(self):
        pass
    
    def update_output2(self):
        name = self.widgets['game_video'].value
        if name is not None:
            with self.out1:
                self.out1.clear_output(True)
                if self.widgets['video'].value:
                    path = os.path.join(CACHE, 'games', name, 'game.mp4')
                    display(Video.from_file(path, play=True, width=800, height=800))
                else:
                    display(self.display_current_image())
    
    def update_output3(self):
        with self.out2:
            self.out2.clear_output(True)
            if self.widgets['summary'].value is not None:
                winners, _ = self.upload_stats()
                display(MultiGame.plot_winners(winners))
    
    def upload_stats(self):
        name = self.widgets['summary'].value
        winners = pd.read_csv(os.path.join(CACHE, 'training_data', name, 'winners.csv'), header=None)
        targets = pd.read_csv(os.path.join(CACHE, 'training_data', name, 'targets.csv'), header=None)
        
        return winners, targets
    
    @staticmethod
    def plot_winners(winners):
        d1 = winners.applymap(lambda x: 'Light' if x == 1 else 'Tie' if x == 0 else 'Dark')
        c = pd.DataFrame(d1.value_counts())
        c['Name'] = list(map(lambda x: x[0], c.index.values))
        c.columns = ['Winner', 'Name']
        return px.pie(c,
                      values='Winner',
                      names='Name',
                      color='Name',
                      color_discrete_map={'Light': 'lightcyan', 'Tie': 'royalblue', 'Dark': 'darkblue'}
                      )
    
    def plot_targets(self):
        pass
    
    def update_output(self):
        self.update_game()
        # self.update_output1()
        self.update_output2()
        self.update_output3()
        possible_rendered = MultiGame.ready_to_render_files()
        self.widgets['game_video'].options = possible_rendered
    
    def display_current_image(self):
        name = self.widgets['game_video'].value
        path = os.path.join(CACHE, 'games', name, 'frames', f'frame{self.image_index}.jpg')
        return Image(filename=path,  width=800, height=800)
    
    def update_game(self):
        if self.player_manager.changed:
            self.game = self.create_game()
    
    def display(self):
        single_game = widgets.VBox([
            widgets.HBox([self.widgets['play_game'], self.widgets['plot']]),
            self.out
        ], layout=_SectionBase.OUT_LAYOUT)
        
        render_game = widgets.VBox([
            widgets.HBox([
                self.widgets['game_video'],
                self.widgets['video'],
            ]),
            widgets.HBox([
                self.widgets['prev'],
                self.widgets['next']
            ]),
            self.out1
        ], layout=_SectionBase.OUT_LAYOUT)

        summary = widgets.VBox([
            widgets.HBox([
                self.widgets['summary'],
            ]),
            self.out2
        ], layout=_SectionBase.OUT_LAYOUT)
        
        out = widgets.Tab([single_game, render_game, summary])
        out.set_title(0, 'Simulate')
        out.set_title(1, 'Render')
        out.set_title(2, 'Stats')
        
        self.update_output()
        return out
    
    def play(self):
        history, winner = self.game.simulate_game()
        return history, winner
    
    def save(self, history):
        path = os.path.join(CACHE, 'games', f'{self.game}-{time.strftime("%b %d %Y %H:%M:%S")}')
        gui.Visualizer.visualize_game(history, path)

    @staticmethod
    def ready_to_summary_files():
        if os.path.exists(os.path.join(CACHE, 'training_data')):
            possible_rendered = list(os.listdir(os.path.join(CACHE, 'training_data')))
            possible_rendered = list(filter(
                lambda x: os.path.exists(os.path.join(CACHE, 'training_data', x, 'winners.csv')),
                possible_rendered
            ))
            possible_rendered = [None] + sorted(possible_rendered)
        else:
            os.makedirs(os.path.join(CACHE, 'training_data'))
            possible_rendered = [None]
        return possible_rendered

    @staticmethod
    def ready_to_render_files():
        if os.path.exists(os.path.join(CACHE, 'games')):
            possible_rendered = list(os.listdir(os.path.join(CACHE, 'games')))
            possible_rendered = list(filter(
                lambda x: os.path.exists(os.path.join(CACHE, 'games', x, 'game.mp4')),
                possible_rendered
            ))
            possible_rendered = [None] + sorted(possible_rendered, key=lambda x: x.split(' ')[-1], reverse=True)
        else:
            os.makedirs(os.path.join(CACHE, 'games'))
            possible_rendered = [None]
        return possible_rendered


if __name__ == '__main__':
    import plotly.express as px
    # pm = PlayerManager()
    # gc = MultiGame(pm)
    name = 'AlphaBetaPlayer(4)vsEpsilonGreedyPlayer'
    d = pd.read_csv(os.path.join(CACHE, 'training_data', name, 'winners.csv'))
    
    print(d)