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
        self.q = np.array([0.0]*n_arms)
        self.Q = np.array([0.0]*n_arms)
        self.Q_reward_sum = {}
        self.Q_action_number = {}
        self.cum_reward = 0.0
        self.num_action = 0
        
        
        for i in xrange(n_arms):
            self.q[i] = np.random.normal() #Gaussian with mean 0.0 and std 1.0
            self.Q_reward_sum[i] = 0.0
            self.Q_action_number[i] = 0
            

    def get_q(self,action):
        assert(action >=0 and action < self.n_arms)
        return self.q[action]
            
    '''Returns reward based on action.
    action is a number in the range[0..n_arms-1]     
    '''    
    def get_reward(self,action):
       q = self.get_q(action)
       return np.random.normal(loc=q,scale=1.0)
       
    def update_Q(self,action, reward):
       self.Q_reward_sum[action] += reward
       self.Q_action_number[action] += 1
       self.Q[action] = self.Q_reward_sum[action]/self.Q_action_number[action]

    '''E gredy selection algorithm, based on Q value estimation
    if E>0 combine exporation and exploitation methods 
    '''
    def select_action(self,E=0.0):
        p = np.random.uniform(0.0,1.0)
        if p<E:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.Q)        
         
    
    '''Returns action reward distribution
    '''        
    def get_reward_dist(self,action,n_trial=1000):
        a = []
        q = self.get_q(action)
        for i in xrange(n_trial):            
            a.append(np.random.normal(loc=q,scale=1.0))
        return a
        
    '''Make one learning step
    and returns average reward, E is e-gready method parameter
    for value function exploring
    '''    
    def make_step(self,E=0.0):
        a = self.select_action(E)
        r = self.get_reward(a)
        self.update_Q(a,r)
        self.cum_reward += r
        self.num_action += 1
        return self.cum_reward / self.num_action
        
        
def make_bandid_test(E=0.0):
    R = np.array([0.0]*1000)
    for t in xrange(2000):
        b = KBandit()
        r = []
        for i in xrange(1000):
            r.append(b.make_step(E))            
        del b        
        R = R + np.array(r)
        del r
    return R / 2000
        
            
        
        
    