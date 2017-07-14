# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
import numpy as np

 
#展示扑克函数
def card_show(cards, info, n):
    
    #扑克牌记录类展示
    if n == 1:
        print info
        names = []
        for i in cards:
            names.append(i.name+i.color)
        print names    
    #Moves展示
    elif n == 2:
        if len(cards) == 0:
            return 0
        print info
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name+j.color)
            moves.append(names)
        print moves    
    #record展示
    elif n == 3:
        print info
        names = []
        for i in cards:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            for j in i[1]:
                tmp_name.append(j.name+j.color)
            tmp.append(tmp_name)
            names.append(tmp)
        print names
        
#在Player的next_moves中选择出牌方法,目前random
def choose(next_move_types, next_moves):
    #要
    if len(next_moves) == 0:
        return "", ""
    else:
        r = np.random.randint(0,len(next_moves))
    return next_move_types[r], next_moves[r]  
    
    
    
    
    
    
    
    
    