# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from game.myclass import Game
from game.rlutil import get_state, get_actions
import game.actions as actions


############################################
#               LR接口类                   #
############################################
class AgentGY(object):
    
    def __init__(self, player=1, models=["rl","random","random"]):
        self.game = None
        self.player = player
        self.models = models
        self.actions_lookuptable = actions.action_dict
        self.dim_actions = len(self.actions_lookuptable) + 1 #不要
        self.dim_states = 30
        
    def reset(self):
        self.game = Game(self.models)
        self.game.game_start()
        return get_state(self.game.playrecords, self.player)
    
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        return get_actions(self.next_moves, self.actions_lookuptable, self.game)
    
    #传入actions的id
    def step(self, action_id):
        action = [self.next_move_types, self.next_moves, action_id]
        self.game.get_next_move(action=action)
        new_state = get_state(self.game.playrecords, self.player)
        
        if self.game.playrecords.winner == 0:
            reward = 0
        elif self.game.playrecords.winner == self.player:
            reward = 1
        else:
            reward = -1
        return  new_state, reward, self.game.end

#rl
if __name__=="__main__":
    agent = AgentGY()
    s = agent.reset()
    print agent.game.get_record().cards_left1
    actions = agent.get_actions_space()
    s_, r, done = agent.step(0)
    print agent.game.get_record().cards_left1
    print agent.game.get_record().records

    