# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
from .myclass import Game
from .rlutil import get_actions
from .actions import  action_dict
from .config import Config

############################################
#               LR接口类                   #
############################################
class Agent(object):
    """
    只可以在player 1进行训练,player 2/3可以random,规则或rl的val_net
    """
    def __init__(self):
        self.game = None
        
        self.my_config = Config()
        self.actions = []

        self.actions_lookuptable = self.my_config.actions_lookuptable
        self.dim_actions = self.my_config.dim_actions
        self.dim_states = self.my_config.dim_states
        
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        self.actions = get_actions(self.next_moves, self.actions_lookuptable, self.game)
        return self.actions
    
    #get_actions_space_state不改变参数
    def get_actions_space_state(self):
        next_move_types, next_moves = self.game.get_next_moves()
        self.actions = get_actions(next_moves, self.actions_lookuptable, self.game)
        return self.actions

    #游戏进行    
    def next_move(self, action):
        self.game.get_next_move(action)
    
    def game_init(self, models=["rl","random","random"], train=True):
        self.models = models
        self.game = Game(self.models, self.my_config)
        self.game.game_start(train)