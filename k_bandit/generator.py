#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:37:26 2017

@author: yonic
"""
import numpy as np

class KBandit(object):
    '''generate k bandit problem generator'''
    def __init__(self,n_arms=10):
        self.n_arms = n_arms
        self.Q = np.array([0.0]*n_arms)
        for i in xrange(n_arms):
            self.Q[i] = np.random.normal() #Gaussian with mean 0.0 and std 1.0
            

    def get_q(self,action):
        assert(action >=0 and action < self.n_arms)
        return self.Q[action]
            
    '''Returns reward based on action.
    action is a number in the range[0..n_arms-1]     
    '''    
    def get_reward(self,action):
       q = self.get_q(action)
       return np.random.normal(loc=q,scale=1.0)
    
    '''Returns action reward distribution
    '''        
    def get_reward_dist(self,action,n_trial=1000):
        a = []
        q = self.get_q(action)
        for i in xrange(n_trial):            
            a.append(np.random.normal(loc=q,scale=1.0))
        return a
            
        
        
    