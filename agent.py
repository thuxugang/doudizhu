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
class Agent(object):
    """
    可以同时兼容训练3个rl,batch
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
        
    def get_actions_space(self, player):
        if player == 1:
            self.s1 = get_state(self.game.playrecords, player)
        elif player == 2:
            self.s2 = get_state(self.game.playrecords, player)
        else:
            self.s3 = get_state(self.game.playrecords, player)
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        self.actions = get_actions(self.next_moves, self.actions_lookuptable, self.game)
        return self.actions
        
    #传入actions的id
    def step(self, player, action_id=0):
        if player == 1:
            s = self.s1
        elif player == 2:
            s = self.s2
        else:
            s = self.s3
        action = [self.next_move_types, self.next_moves, action_id, self.actions]
        rl_record = RLRecord(s=s, a=self.actions[action_id])
        done = self.game.play(rl_record, action=action)
        return done
    
    #获取训练样本
    def get_training_data(self):
        return self.game.q1, self.game.q2, self.game.q3
     
#rl
if __name__=="__main__":
    agent = Agent(models=["random","random","random"])
    agent.reset()
    done = False
    while(True):
        print(agent.game.get_record().cards_left1)
        actions = agent.get_actions_space(player=1)
        #choose action_id
        done = agent.step(player=1, action_id=0)
        if done:
            break
        print(agent.game.get_record().cards_left1)
        print(agent.game.get_record().cards_left2)
        print(agent.game.get_record().cards_left3)
        print(agent.game.get_record().records)

        actions = agent.get_actions_space(player=2)
        #choose action_id
        done = agent.step(player=2, action_id=0)
        if done:
            break
        actions = agent.get_actions_space(player=3)
        #choose action_id
        done = agent.step(player=3, action_id=0)
        if done:
            break        
        #print("====================")  
    
        #每轮更新方法[-1],返回为LR记录类对象列表
        d1, d2, d3 = agent.get_training_data()
        print(len(d1),len(d2),len(d3))
        
    #回合更新方法，返回为LR记录类对象列表
    d1, d2, d3 = agent.get_training_data()
    print(len(d1),len(d2),len(d3))


    