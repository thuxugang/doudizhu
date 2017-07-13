# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: xu
"""
from myclass import Cards, Player, PlayRecords
from game_process import game_start
from myutil import card_show
   
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
    playrecords.show("start")
    
    players[0].go()
    players[0].total_moves.show('1')
    
    
    
    
    
    
    
    
    