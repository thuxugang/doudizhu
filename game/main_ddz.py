# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
from .myclass import Cards, Player, PlayRecords, WebShow
from .myutil import game_init
import jsonpickle

                    
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
        
    #游戏进行    
    def next_move(self):
        
        self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.i].go(self.last_move_type, self.last_move, self.playrecords, self.model)
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
        
   
if __name__=="__main__":
    
    game_ddz = Game("random")  
    game_ddz.game_start()
    
    i = 0
    while(game_ddz.playrecords.winner == 0):
        game_ddz.playrecords.show(str(i))
        game_ddz.next_move()
        i = i + 1

    
    
    