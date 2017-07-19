# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from game.myclass import Cards, Player, PlayRecords, WebShow
from game.myutil import game_init, get_state, get_actions
import jsonpickle
import game.actions as actions
import numpy as np

############################################
#                 游戏类                   #
############################################                   
class Game(object):
    
    def __init__(self, model):
        #初始化一副扑克牌类
        self.cards = Cards()
        
        #play相关参数
        self.end = False
        self.last_move_type = self.last_move = "start"
        self.playround = 1
        self.i = 0
        self.yaobuqis = []
        
        #choose模型
        self.model = model
        
    #发牌
    def game_start(self):
        
        #初始化players
        self.players = []
        for i in range(1,4):
            self.players.append(Player(i))
        
        #初始化扑克牌记录类
        self.playrecords = PlayRecords()    
        
        #发牌
        game_init(self.players, self.playrecords, self.cards)
    
    
    #返回扑克牌记录类
    def get_record(self):
        web_show = WebShow(self.playrecords)
        return jsonpickle.encode(web_show, unpicklable=False)
    
    #返回下次出牌列表
    def get_next_moves(self):
        next_move_types, next_moves = self.players[self.i].get_moves(self.last_move_type, self.last_move, self.playrecords)
        return next_move_types, next_moves
    
    #游戏进行    
    def next_move(self, action):
        self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.i].go(self.last_move_type, self.last_move, self.playrecords, self.model, action)
        if self.yaobuqi:
            self.yaobuqis.append(self.i)
        else:
            self.yaobuqis = []
        #都要不起
        if len(self.yaobuqis) == 2:
            self.yaobuqis = []
            self.last_move_type = self.last_move = "start"
        if self.end:
            self.playrecords.winner = self.i+1
        self.i = self.i + 1
        #一轮结束
        if self.i > 2:
            #playrecords.show("=============Round " + str(playround) + " End=============")
            self.playround = self.playround + 1
            #playrecords.show("=============Round " + str(playround) + " Start=============")
            self.i = 0 

############################################
#               LR接口类                   #
############################################
class AgentGY:
    
    def __init__(self):
        self.game = None
        self.player1 = "rl"
        self.player2 = "random"
        self.player3 = "random"
        self.actions_lookuptable = actions.action_dict
        
    def reset(self):
        self.game = Game("rl")
        self.game.game_start()
        return get_state(self.game.playrecords, 1)
    
    def get_actions_space(self):
        self.next_move_types, self.next_moves = self.game.get_next_moves()
        return get_actions(self.next_moves, self.actions_lookuptable, self.game)
    
    #传入actions的id
    def step(self, action_id):
        action = [self.next_move_types, self.next_moves, action_id]
        self.game.next_move(action=action)
        new_state = get_state(self.game.playrecords, 1)
        
        if self.game.playrecords.winner == 0:
            reward = 0
        elif self.game.playrecords.winner == 1:
            reward = 1
        else:
            reward = -1
        return  new_state, reward, self.game.end

#rl
if __name__=="__main__":
    agent = AgentGY()
    s = agent.reset()
    actions = agent.get_actions_space()
    s_, r, done = agent.step(0)
        
"""  
if __name__=="__main__":
    
    game_ddz = Game("random")  
    game_ddz.game_start()
    
    i = 0
    while(game_ddz.playrecords.winner == 0):
        game_ddz.get_next_moves()
        game_ddz.next_move()
        game_ddz.playrecords.show(str(i))
        print game_ddz.playrecords.player
        i = i + 1
"""
    
    
    