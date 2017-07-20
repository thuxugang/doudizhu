# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
import numpy as np
 
#展示扑克函数
def card_show(cards, info, n):
    
    #扑克牌记录类展示
    if n == 1:
        print(info)
        names = []
        for i in cards:
            names.append(i.name+i.color)
        print(names)  
    #Moves展示
    elif n == 2:
        if len(cards) == 0:
            return 0
        print(info)
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name+j.color)
            moves.append(names)
        print(moves)    
    #record展示
    elif n == 3:
        print(info)
        names = []
        for i in cards:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            #处理要不起
            try:
                for j in i[1]:
                    tmp_name.append(j.name+j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            names.append(tmp)
        print(names)
       

#在Player的next_moves中选择出牌方法
def choose(next_move_types, next_moves, last_move_type, model, action):
    
    if model == "random":
        return choose_random(next_move_types, next_moves, last_move_type)
    elif model == "rl":
        if action[3][action[2]] == 429:
            return "buyao", []
        elif action[3][action[2]] == 430:
            return "yaobuqi", []
        else:
            return action[0][action[2]], action[1][action[2]] 
        
#random
def choose_random(next_move_types, next_moves, last_move_type):
    #要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    else:
        #start不能不要
        if last_move_type == "start":
            r_max = len(next_moves)
        else:
            r_max = len(next_moves)+1
        r = np.random.randint(0,r_max)
        #添加不要
        if r == len(next_moves):
            return "buyao", []
        
    return next_move_types[r], next_moves[r] 
    
#发牌
def game_init(players, playrecords, cards):
    
    #洗牌
    np.random.shuffle(cards.cards)
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
    