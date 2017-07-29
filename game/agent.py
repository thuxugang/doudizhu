# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
from .myclass import Game
from .rlutil import get_state, get_actions, combine


############################################
#               LR接口类                   #
############################################
class Agent(object):
    """
    只可以在player 1进行训练,player 2/3可以random,规则或rl的val_net
    """
    def __init__(self, player=1, models=["rl","random","random"], my_config=None, RL=None, train=True):
        self.game = None
        self.player = player
        self.models = models
        self.actions_lookuptable = my_config.actions_lookuptable
        self.dim_actions = my_config.dim_actions
        self.dim_states = my_config.dim_states
        
        self.actions = []
        
        self.train = train
        
        self.RL = RL
        
    def reset(self):
        self.game = Game(self, self.RL)
        self.game.game_start(self.train)
        return get_state(self.game.playrecords, self.player)
    
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        self.actions = get_actions(self.next_moves, self.actions_lookuptable, self.game)
        return self.actions
    
    #get_actions_space_state不改变参数
    def get_actions_space_state(self):
        next_move_types, next_moves = self.game.get_next_moves()
        self.actions = get_actions(next_moves, self.actions_lookuptable, self.game)
        return self.actions
        
    #传入actions的id
    def step(self, action_id=0):
        action = [self.next_move_types, self.next_moves, action_id, self.actions]
        winner, done = self.game.get_next_move(action=action)
        new_state = get_state(self.game.playrecords, self.player)
        
        
        if winner == 0:
            #不出reward -0.1
            if self.actions[action_id] == 429:
                reward = -0.1
            else:
                reward = 0
        elif winner == self.player:
            reward = 1
        else:
            reward = -1
        return  new_state, reward, done

#rl
if __name__=="__main__":
    agent = Agent()
    s = agent.reset()
    done = False
    while(not done):
        actions = agent.get_actions_space() #如果actions为[]，step()
        s = combine(s, actions)
        print(actions)
        #GY的RL程序
        s_, r, done = agent.step(action_id=0)
        
        actions_ = agent.get_actions_space_state()
        print(actions_)
        s_ = combine(s_, actions_) #get_actions_space_state不改变game参数
        print("====================")       
        raw_input("")
        s = s_


    