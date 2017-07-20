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
class Agent(object):
    """
    只可以在player 1进行训练,player 2/3可以random,规则或rl的val_net
    """
    def __init__(self, player=1, models=["rl","random","random"]):
        self.game = None
        self.player = player
        self.models = models
        self.actions_lookuptable = actions.action_dict
        self.dim_actions = len(self.actions_lookuptable) + 2 #429 buyao, 430 yaobuqi
        self.dim_states = 30 + 3
        
        self.actions = []
        
    def reset(self):
        self.game = Game(self.models)
        self.game.game_start()
        return get_state(self.game.playrecords, self.player)
    
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
    agent = Agent()
    s = agent.reset()
    done = False
    while(not done):
        print(agent.game.get_record().cards_left1)
        actions = agent.get_actions_space() #如果actions为[]，step()
        #GY的RL程序
        s_, r, done = agent.step(action_id=0)
        print(agent.game.get_record().cards_left1)
        print(agent.game.get_record().cards_left2)
        print(agent.game.get_record().cards_left3)
        print(agent.game.get_record().records)
        print("====================")       
        #raw_input("")
        s = s_


    