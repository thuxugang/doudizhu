# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
import numpy as np
from myutil import card_show


############################################
#              扑克牌相关类                 #
############################################
class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """
    def __init__(self,):
        #初始化扑克牌类型
        self.cards_type = ['1-a-12', '1-b-12','1-c-12','1-d-12',
                           '2-a-13', '2-b-13','2-c-13','2-d-13',
                           '3-a-1', '3-b-1','3-c-1','3-d-1',
                           '4-a-2', '4-b-2','4-c-2','4-d-2',
                           '5-a-3', '5-b-3','5-c-3','5-d-3',
                           '6-a-4', '6-b-4','6-c-4','6-d-4',
                           '7-a-5', '7-b-5','7-c-5','7-d-5',
                           '8-a-6', '8-b-6','8-c-6','8-d-6',
                           '9-a-7', '9-b-7','9-c-7','9-d-7',
                           '10-a-8', '10-b-8','10-c-8','10-d-8',
                           '11-a-9', '11-b-9','11-c-9','11-d-9',
                           '12-a-10', '12-b-10','12-c-10','12-d-10',
                           '13-a-11', '13-b-11','13-c-11','13-d-11',
                           '14-a-14', '15-a-15']
        #初始化扑克牌类                  
        self.cards = self.get_cards()

    #初始化扑克牌类
    def get_cards(self):
        cards = []
        for card_type in self.cards_type:
            cards.append(Card(card_type))
        #打乱顺序
        np.random.shuffle(cards)
        return cards
    
                           
class Card(object):
    """
    扑克牌类
    """
    def __init__(self, card_type):
        self.card_type = card_type
        #名称
        self.name = self.card_type.split('-')[0]
        #花色
        self.color = self.card_type.split('-')[1]
        #大小
        self.rank = int(self.card_type.split('-')[2])
        
    #判断大小
    def bigger_than(self, card_instance):
        if (self.rank > card_instance.rank):
            return True
        else:
            return False
     
class PlayRecords(object):
    """
    扑克牌记录类
    """
    def __init__(self):
        self.cards_left1 = []
        self.cards_left2 = []
        self.cards_left3 = []
        
        #出牌记录
        self.records = []
    
    #展示
    def show(self, info):
        print info
        card_show(self.cards_left1, "player 1", 1)
        card_show(self.cards_left2, "player 2", 1)
        card_show(self.cards_left3, "player 3", 1)
        print "records"
        print self.records


############################################
#              出牌相关类                   #
############################################
class Moves(object):
    """
    出牌类,单,对,三,三带一,三带二,顺子,炸弹
    """ 
    def __init__(self):
        #出牌信息
        self.dan = []
        self.dui = []
        self.san = []
        self.san_dai_yi = []
        self.san_dai_er = []
        self.bomb = []
        self.shunzi = []

        #牌数量信息
        self.card_num_info = {}
        #牌顺序信息,计算顺子
        self.card_order_info = []
        
    #获取全部出牌列表
    def get_moves(self, cards_left):
        
        #统计牌数量/顺序信息
        for i in cards_left:
            #数量
            tmp = self.card_num_info.get(i.name, [])
            if len(tmp) == 0:
                self.card_num_info[i.name] = [i]
            else:
                self.card_num_info[i.name].append(i)
            #顺序
            if i.rank in [13,14,15]: #不统计2,小王,大王
                continue
            elif len(self.card_order_info) == 0:
                self.card_order_info.append(i)
            elif i.rank != self.card_order_info[-1].rank:
                self.card_order_info.append(i)
                
        #出单,出对,出三,炸弹(考虑拆开)
        for k, v in self.card_num_info.items():
            if len(v) == 1:
                self.dan.append(v)
            elif len(v) == 2:
                self.dui.append(v)
                self.dan.append(v[:1])
            elif len(v) == 3:
                self.san.append(v)
                self.dui.append(v[:2])
                self.dan.append(v[:1])
            elif len(v) == 4:
                self.bomb.append(v)
                self.san.append(v[:3])
                self.dui.append(v[:2])
                self.dan.append(v[:1])
                
        #三带一,三带二
        for san in self.san:
            for dan in self.dan:
                #防止重复
                if dan[0].name != san[0].name:
                    self.san_dai_yi.append(san+dan)
            for dui in self.dui:
                #防止重复
                if dui[0].name != san[0].name:
                    self.san_dai_er.append(san+dui)  
                    
        #获取最长顺子
        max_len = []
        for i in self.card_order_info:
            if i == self.card_order_info[0]:
                max_len.append(i)
            elif max_len[-1].rank == i.rank - 1:
                max_len.append(i)
            else:
                if len(max_len) >= 5:
                   self.shunzi.append(max_len) 
                max_len = [i]
        #最后一轮
        if len(max_len) >= 5:
           self.shunzi.append(max_len)   
        #拆顺子 
        shunzi_sub = []             
        for i in self.shunzi:
            len_total = len(i)
            n = len_total - 5
            #遍历所有可能顺子长度
            while(n > 0):
                len_sub = len_total - n
                j = 0
                while(len_sub+j <= len(i)):
                    #遍历该长度所有组合
                    shunzi_sub.append(i[j:len_sub+j])
                    j = j + 1
                n = n - 1
        self.shunzi.extend(shunzi_sub)
                
            
            
            
            
    #展示
    def show(self, info):
        print info
        card_show(self.dan, "dan", 2)
        card_show(self.dui, "dui", 2)
        card_show(self.san, "san", 2)
        card_show(self.san_dai_yi, "san_dai_yi", 2)
        card_show(self.san_dai_er, "san_dai_er", 2)
        card_show(self.bomb, "bomb", 2)
        card_show(self.shunzi, "shunzi", 2)


############################################
#              玩家相关类                   #
############################################        
class Player(object):
    """
    player类
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.cards_left = []
        #所有出牌可选列表
        self.total_moves = Moves()
        #下次出牌可选列表
        self.next_moves = Moves()
        
    
    def go(self):
    #def go(self, last_move_type, last_move):
        #获取全部出牌列表
        self.total_moves.get_moves(self.cards_left)

    
    
    
    
    