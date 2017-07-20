# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
from .gameutil import card_show, choose, game_init
from .rlutil import get_state


############################################
#                 LR记录类                  #
############################################  
class RLRecord(object):
    def __init__(self, s=None, a=None, r=None,s_=None):
        self.s = s
        self.a = a
        self.r = r
        self.s_ = s_
        
############################################
#                 游戏类                   #
############################################                   
class Game(object):
    
    def __init__(self, models):
        #初始化一副扑克牌类
        self.cards = Cards()
        
        #play相关参数
        self.end = False
        self.last_move_type = self.last_move = "start"
        self.playround = 1
        self.i = 0
        self.yaobuqis = []
        
        #choose模型
        self.models = models
        
        #[s,a,r,s_]
        self.q1 = []
        self.q2 = []
        self.q3 = []
        
        #记录rlrecord
        self.rlrecord1 = None
        self.rlrecord2 = None
        self.rlrecord3 = None
        
    #发牌
    def game_start(self):
        
        #初始化players
        self.players = []
        self.players.append(Player(1, self.models[0]))
        self.players.append(Player(2, self.models[1]))
        self.players.append(Player(3, self.models[2]))
        
        #初始化扑克牌记录类
        self.playrecords = PlayRecords()    
        
        #发牌
        game_init(self.players, self.playrecords, self.cards)
    
    #返回扑克牌记录类
    def get_record(self):
        web_show = WebShow(self.playrecords)
        #return jsonpickle.encode(web_show, unpicklable=False)
        return web_show
    
    #返回下次出牌列表
    def get_next_moves(self):
        next_move_types, next_moves = self.players[self.i].get_moves(self.last_move_type, self.last_move, self.playrecords)
        return next_move_types, next_moves
    
        
    #游戏进行    
    def play(self, rl_record, action):

        #记录rl_record
        if self.playround > 1:
            if self.i == 0:
                self.q1.append(RLRecord(s=self.rlrecord1.s, a=self.rlrecord1.a, r=0, s_=get_state(self.playrecords, 1)))
            elif self.i == 1:
                self.q2.append(RLRecord(s=self.rlrecord2.s, a=self.rlrecord2.a, r=0, s_=get_state(self.playrecords, 2)))
            else:
                self.q3.append(RLRecord(s=self.rlrecord3.s, a=self.rlrecord3.a, r=0, s_=get_state(self.playrecords, 3)))   
        
        if self.i == 0:
            self.rlrecord1 = rl_record
        elif self.i == 1:
            self.rlrecord2 = rl_record
        else:
            self.rlrecord3 = rl_record            
                
        self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.i].play(self.last_move_type, self.last_move, self.playrecords, action)
        
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
            #记录rl_record
            if self.playrecords.winner == 1:
                self.q1.append(RLRecord(s=self.rlrecord1.s, a=self.rlrecord1.a, r=1, s_=get_state(self.playrecords, 1)))
                self.q2.append(RLRecord(s=self.rlrecord2.s, a=self.rlrecord2.a, r=-1, s_=get_state(self.playrecords, 2)))
                self.q3.append(RLRecord(s=self.rlrecord3.s, a=self.rlrecord3.a, r=-1, s_=get_state(self.playrecords, 3)))
            elif self.playrecords.winner == 2:
                self.q1.append(RLRecord(s=self.rlrecord1.s, a=self.rlrecord1.a, r=-1, s_=get_state(self.playrecords, 1)))
                self.q2.append(RLRecord(s=self.rlrecord2.s, a=self.rlrecord2.a, r=1, s_=get_state(self.playrecords, 2)))
                self.q3.append(RLRecord(s=self.rlrecord3.s, a=self.rlrecord3.a, r=-1, s_=get_state(self.playrecords, 3)))
            else:
                self.q1.append(RLRecord(s=self.rlrecord1.s, a=self.rlrecord1.a, r=-1, s_=get_state(self.playrecords, 1)))
                self.q2.append(RLRecord(s=self.rlrecord2.s, a=self.rlrecord2.a, r=-1, s_=get_state(self.playrecords, 2)))
                self.q3.append(RLRecord(s=self.rlrecord3.s, a=self.rlrecord3.a, r=1, s_=get_state(self.playrecords, 3)))
            return self.end
            
        self.i = self.i + 1
        
        #一轮结束
        if self.i > 2:
            self.playround = self.playround + 1
            print("================ " + str(self.playround) + " ================")
            self.i = 0 
        
        return self.end
            
############################################
#              扑克牌相关类                 #
############################################
class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """
    def __init__(self):
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
        #np.random.shuffle(cards)
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
        #当前手牌
        self.cards_left1 = []
        self.cards_left2 = []
        self.cards_left3 = []
        
        #可能出牌选择
        self.next_moves1 = []
        self.next_moves2 = []
        self.next_moves3 = []

        #出牌记录
        self.next_move1 = []
        self.next_move2 = []
        self.next_move3 = []
        
        #出牌记录
        self.records = []
        
        #胜利者
        #winner=0,1,2,3 0表示未结束,1,2,3表示winner
        self.winner = 0
        
        #出牌者
        self.player = 1
   
    #展示
    def show(self, info):
        print(info)
        card_show(self.cards_left1, "player 1", 1)
        card_show(self.cards_left2, "player 2", 1)
        card_show(self.cards_left3, "player 3", 1)
        card_show(self.records, "record", 3)


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
        #王牌信息
        self.king = []
        
        #下次出牌
        self.next_moves = []
        #下次出牌类型
        self.next_moves_type = []
        
    #获取全部出牌列表
    def get_total_moves(self, cards_left):
        
        #统计牌数量/顺序/王牌信息
        for i in cards_left:
            #王牌信息
            if i.rank in [14,15]:
                self.king.append(i)
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
        
        #王炸
        if len(self.king) == 2:
            self.bomb.append(self.king)
            
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
                
    #获取下次出牌列表
    def get_next_moves(self, last_move_type, last_move): 
        #没有last,全加上,除了bomb最后加
        if last_move_type == "start":
            moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]
            i = 0
            for move_type in [self.dan, self.dui, self.san, self.san_dai_yi, 
                      self.san_dai_er, self.shunzi]:
                for move in move_type:
                    self.next_moves.append(move)
                    self.next_moves_type.append(moves_types[i])
                i = i + 1
        #出单
        elif last_move_type == "dan":
            for move in self.dan:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)  
                    self.next_moves_type.append("dan")
        #出对
        elif last_move_type == "dui":
            for move in self.dui:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("dui")
        #出三个
        elif last_move_type == "san":
            for move in self.san:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("san")
        #出三带一
        elif last_move_type == "san_dai_yi":
            for move in self.san_dai_yi:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)    
                    self.next_moves_type.append("san_dai_yi")
        #出三带二
        elif last_move_type == "san_dai_er":
            for move in self.san_dai_er:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)   
                    self.next_moves_type.append("san_dai_er")
        #出炸弹
        elif last_move_type == "bomb":
            for move in self.bomb:
                #比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move) 
                    self.next_moves_type.append("bomb")
        #出顺子
        elif last_move_type == "shunzi":
            for move in self.shunzi:
                #相同长度
                if len(move) == len(last_move):
                    #比last大
                    if move[0].bigger_than(last_move[0]):
                        self.next_moves.append(move) 
                        self.next_moves_type.append("shunzi")
        else:
            print("last_move_type_wrong")
            
        #除了bomb,都可以出炸
        if last_move_type != "bomb":
            for move in self.bomb:
                self.next_moves.append(move) 
                self.next_moves_type.append("bomb")
                
        return self.next_moves_type, self.next_moves
    
    
    #展示
    def show(self, info):
        print(info)
        #card_show(self.dan, "dan", 2)
        #card_show(self.dui, "dui", 2)
        #card_show(self.san, "san", 2)
        #card_show(self.san_dai_yi, "san_dai_yi", 2)
        #card_show(self.san_dai_er, "san_dai_er", 2)
        #card_show(self.bomb, "bomb", 2)
        #card_show(self.shunzi, "shunzi", 2)
        #card_show(self.next_moves, "next_moves", 2)


############################################
#              玩家相关类                   #
############################################        
class Player(object):
    """
    player类
    """
    def __init__(self, player_id, model):
        self.player_id = player_id
        self.cards_left = []
        #出牌模式
        self.model = model

    #展示
    def show(self, info):
        self.total_moves.show(info)
        card_show(self.next_move, "next_move", 1)
        #card_show(self.cards_left, "card_left", 1)
        
    #根据next_move同步cards_left
    def record_move(self, playrecords):
        #记录出牌者
        playrecords.player = self.player_id
        #playrecords中records记录[id,next_move]
        if self.next_move_type in ["yaobuqi", "buyao"]:
            self.next_move = self.next_move_type
            playrecords.records.append([self.player_id, self.next_move_type])
        else:
            playrecords.records.append([self.player_id, self.next_move])
            for i in self.next_move:
               self.cards_left.remove(i) 
        #同步playrecords
        if self.player_id == 1:
            playrecords.cards_left1 = self.cards_left
            playrecords.next_moves1.append(self.next_moves)
            playrecords.next_move1.append(self.next_move)
        elif self.player_id == 2:
            playrecords.cards_left2 = self.cards_left 
            playrecords.next_moves2.append(self.next_moves)
            playrecords.next_move2.append(self.next_move)
        elif self.player_id == 3:
            playrecords.cards_left3 = self.cards_left  
            playrecords.next_moves3.append(self.next_moves)
            playrecords.next_move3.append(self.next_move)
        #是否牌局结束
        end = False
        if len(self.cards_left) == 0:
            end = True
        return end
    
    #选牌
    def get_moves(self, last_move_type, last_move, playrecords):
        #所有出牌可选列表
        self.total_moves = Moves()
        #获取全部出牌列表
        self.total_moves.get_total_moves(self.cards_left)
        #获取下次出牌列表
        self.next_move_types, self.next_moves = self.total_moves.get_next_moves(last_move_type, last_move)        
        #返回下次出牌列表
        return self.next_move_types, self.next_moves
        
    #出牌
    def play(self, last_move_type, last_move, playrecords, action):
        #在next_moves中选择出牌方法
        self.next_move_type, self.next_move = choose(self.next_move_types, self.next_moves, last_move_type, self.model, action)
        #记录
        end = self.record_move(playrecords)
        #展示
        #self.show("Player " + str(self.player_id))  
        #要不起&不要
        yaobuqi = False
        if self.next_move_type in ["yaobuqi","buyao"]:
            yaobuqi = True
            self.next_move_type = last_move_type
            self.next_move = last_move
            
        return self.next_move_type, self.next_move, end, yaobuqi
    
    
############################################
#               网页展示类                 #
############################################
class WebShow(object):
    """
    网页展示类
    """    
    def __init__(self, playrecords):
        
        #胜利者
        self.winner = playrecords.winner
        
        #剩余手牌
        self.cards_left1 = []
        for i in playrecords.cards_left1:
            self.cards_left1.append(i.name+i.color)
        self.cards_left2 = []
        for i in playrecords.cards_left2:
            self.cards_left2.append(i.name+i.color)        
        self.cards_left3 = []
        for i in playrecords.cards_left3:
            self.cards_left3.append(i.name+i.color)        
        
        #可能出牌
        self.next_moves1 = []
        if len(playrecords.next_moves1) != 0:
            next_moves = playrecords.next_moves1[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name+card.color)  
                self.next_moves1.append(cards)
        self.next_moves2 = []
        if len(playrecords.next_moves2) != 0:
            next_moves = playrecords.next_moves2[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name+card.color)  
                self.next_moves2.append(cards)        
        self.next_moves3 = []
        if len(playrecords.next_moves3) != 0:
            next_moves = playrecords.next_moves3[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name+card.color)  
                self.next_moves3.append(cards)   
                
        #出牌
        self.next_move1 = []
        if len(playrecords.next_move1) != 0:
            next_move = playrecords.next_move1[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move1.append(next_move)
            else:
                for card in next_move:
                    self.next_move1.append(card.name+card.color)  
        self.next_move2 = []
        if len(playrecords.next_move2) != 0:
            next_move = playrecords.next_move2[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move2.append(next_move)
            else:
                for card in next_move:
                    self.next_move2.append(card.name+card.color) 
        self.next_move3 = []
        if len(playrecords.next_move3) != 0:
            next_move = playrecords.next_move3[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move3.append(next_move)
            else:
                for card in next_move:
                    self.next_move3.append(card.name+card.color) 
                
        #记录
        self.records = []
        for i in playrecords.records:
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
            self.records.append(tmp)        

