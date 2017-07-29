# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from .actions import  action_dict


############################################
#                  config                  #
############################################
class Config(object):
    def __init__(self):
        self.actions_lookuptable = action_dict
        self.dim_actions = len(self.actions_lookuptable) + 2 #429 buyao, 430 yaobuqi
        self.dim_states = 30 + 3 + 431 #431为dim_actions
