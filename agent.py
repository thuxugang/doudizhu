# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from game.myclass import Game
from game.rlutil import get_state, get_actions
import game.actions as actions


############################################
#               LR接口类                   #
############################################
class Agents(object):
    """
    可以同时兼容训练3个
    """
    def __init__(self, models=["rl","rl","rl"]):
        self.game = None
        self.models = models
        self.actions_lookuptable = actions.action_dict
        self.dim_actions = len(self.actions_lookuptable) + 2 #429 buyao, 430 yaobuqi
        self.dim_states = 30 + 3
        
        self.actions = []
        
    def reset(self):
        self.game = Game(self.models)
        self.game.game_start()
        
        self.agent1 = Agent(player=1, game=self.game, actions_lookuptable=self.actions_lookuptable)
        self.agent2 = Agent(player=2, game=self.game, actions_lookuptable=self.actions_lookuptable)
        self.agent3 = Agent(player=3, game=self.game, actions_lookuptable=self.actions_lookuptable)
            
        return get_state(self.game.playrecords, 1), get_state(self.game.playrecords, 2), get_state(self.game.playrecords, 3)
    
    
class Agent(object):
    """
    每一个player类
    """
    def __init__(self, player=1,  game=None, actions_lookuptable=None):
        self.game = game
        self.player = player
        self.actions_lookuptable = actions_lookuptable
        self.actions = []
    
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        self.actions = get_actions(self.next_moves, self.actions_lookuptable, self.game)
        return self.actions
        
    #传入actions的id
    def step(self, action_id=0):
        action = [self.next_move_types, self.next_moves, action_id, self.actions]
        winner, done = self.game.get_next_move(action=action)
        new_state = get_state(self.game.playrecords, self.player)
        
        if winner == 0:
            reward = 0
        elif winner == self.player:
            reward = 1
        else:
            reward = -1
        return  new_state, reward, done

#rl
if __name__=="__main__":
    agents = Agents()
    s1, s2, s3 = agents.reset()
    done = False
    while(not done):
        print(agents.game.get_record().cards_left1)
        actions = agents.agent1.get_actions_space()
        #GY的RL程序
        s_, r, done = agents.agent1.step(action_id=0)
        print(agents.game.get_record().cards_left1)
        print(agents.game.get_record().cards_left2)
        print(agents.game.get_record().cards_left3)
        print(agents.game.get_record().records)
        print("====================")       
        #raw_input("")
        s = s_


    