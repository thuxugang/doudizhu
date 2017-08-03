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
from rl.init_model import model_init

############################################
#               LR接口类                   #
############################################
class Agent(object):
    """
    只可以在player 1进行训练,player 2/3可以random,规则或rl的val_net
    """
    def __init__(self):
        self.game = None
        
        self.actions_lookuptable = action_dict
        self.dim_actions = len(self.actions_lookuptable) + 2 #429 buyao, 430 yaobuqi
        self.dim_states = 30 + 3 + 431 #431为dim_actions
        
        self.actions = []
        self.RL = model_init(self, rl_model="prioritized_dqn", e_greedy=1, start_iter=500000)
        
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
    def next_move(self):
        self.game.get_next_move()
    
    def game_init(self, models=["rl","random","random"], train=True):
        self.models = models
        self.game = Game(self, self.RL)
        self.game.game_start(train)