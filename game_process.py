# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: xu
"""

#发牌
def game_init(players, playrecords, cards):
    
    #排序
    p1_cards = cards.cards[:18]
    p1_cards.sort(key=lambda x: x.rank)
    p2_cards = cards.cards[18:36]
    p2_cards.sort(key=lambda x: x.rank)
    p3_cards = cards.cards[36:]
    p3_cards.sort(key=lambda x: x.rank)
    players[0].cards_left = playrecords.cards_left1 = p1_cards
    players[1].cards_left = playrecords.cards_left2 = p2_cards
    players[2].cards_left = playrecords.cards_left3 = p3_cards

    
#游戏进行    
def game_start(players, playrecords):
    
    end = False
    last_move_type = last_move = "start"
    playround = 1
    i = 0
    yaobuqis = []
    while(True):
        last_move_type, last_move, end, yaobuqi = players[i].go(last_move_type, last_move, playrecords, str(i+1))
        if yaobuqi:
            yaobuqis.append(i)
        else:
            yaobuqis = []
        #都要不起
        if len(yaobuqis) == 2:
            yaobuqis = []
            last_move_type = last_move = "start"
        if end:
            winner = str(i+1)
            break
        i = i + 1
        #一轮结束
        if i > 2:
            playrecords.show("=============Round " + str(playround) + " End=============")
            playround = playround + 1
            playrecords.show("=============Round " + str(playround) + " Start=============")
            i = 0    
    
    return winner
    
    
    
    
    
    