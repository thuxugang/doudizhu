# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from game.myclass import Game, RLRecord
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
            
 
class Agent(object):
    """
    每一个player类
    """
    def __init__(self, player,  game=None, actions_lookuptable=None):
        self.game = game
        self.player = player
        self.actions_lookuptable = actions_lookuptable
        self.actions = []
        self.rl_rs = []
    
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        self.actions = get_actions(self.next_moves, self.actions_lookuptable, self.game)
        return self.actions
        
    #传入actions的id
    def step(self, s, action_id=0):
        action = [self.next_move_types, self.next_moves, action_id, self.actions]
        rl_record = RLRecord(s=s, a=self.actions[action_id])
        done = self.game.play(rl_record, action=action)
        return  done

#rl
if __name__=="__main__":
    agents = Agents(models=["rl","rl","rl"])
    agents.reset()
    done = False
    while(True):
        s1 = get_state(agents.game.playrecords, 1)
        print(agents.game.get_record().cards_left1)
        actions = agents.agent1.get_actions_space()
        #GY的RL程序
        done = agents.agent1.step(s1)
        if done:
            break
        print(agents.game.get_record().cards_left1)
        print(agents.game.get_record().cards_left2)
        print(agents.game.get_record().cards_left3)
        print(agents.game.get_record().records)

        s2 = get_state(agents.game.playrecords, 2)
        actions = agents.agent2.get_actions_space()
        done = agents.agent2.step(s2)
        if done:
            break
        s3 = get_state(agents.game.playrecords, 3)
        actions = agents.agent3.get_actions_space()
        done = agents.agent3.step(s3)
        if done:
            break        
        print("====================")  



    