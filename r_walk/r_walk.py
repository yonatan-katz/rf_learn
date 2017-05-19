#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:07:47 2017

@author: yonic
"""
import numpy as np
import numpy.random as random

class RWalk:
    def __init__(self):
        self.states = ['A','B','C','D','E']
        self.V = np.array([0.5*len(self.states)])
        self.state = 2
        self.
    
    '''Returns reward and True is eposode is finish'''
    def one_step(self):
        p = random.uniform()
        if p<0.5:
            self.state -= 1
            if self.state == -1:
                return 0,True #stop episode
            elif self
            
            
        
        