# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: xu
"""
from myclass import Cards, Player, PlayRecords
from game_process import game_start
   
    
if __name__=="__main__":
    
    #初始化一副扑克牌类
    cards = Cards()
    
    #初始化players
    players = []
    for i in range(1,4):
        players.append(Player(i))
    
    #初始化扑克牌记录类
    playrecords = PlayRecords()
    
    #发牌
    game_start(players, playrecords, cards)
    playrecords.show("=============START=============")
    
    end = False
    last_move_type = last_move = ""
    playround = 0
    i = 0
    while(not end):
        last_move_type, last_move, end = players[i].go(last_move_type, last_move, playrecords, str(i+1))
        i = i + 1
        if i > 2:
            playround = playround + 1
            playrecords.show("=============Round " + str(playround) + "=============")
            i = 0
    
    
    
    
    
    
    
    
    