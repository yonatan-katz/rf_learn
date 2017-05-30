#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:07:47 2017

@author: yonic
"""
import numpy as np
import numpy.random as random
from collections import OrderedDict

class RWalk:
    def __init__(self):
        self.states = ['term','A','B','C','D','E','term']
        self.v = {'A':1.0/6,'B':2.0/6,'C':3.0/6,'D':4.0/6,'E':5.0/6}
        self.state = 3       
        
    def start_new_episode(self):
        self.state = 3
        return self.states[self.state]
    
    '''Returns reward and True is eposode is finish'''
    def one_step(self):
        p = random.uniform()
        if p<0.5:
            self.state -= 1
            if self.state == 0:
                return 0,self.states[self.state],True #stop episode
            else:
                return 0,self.states[self.state],False
        else:
            self.state += 1
            if self.state == 6:
                return 1,self.states[self.state],True
            else:
                return 0,self.states[self.state],False


def flat_dict(d):
    v = []
    for k in sorted(d.keys()):
        v.append(d[k])
    return v
    
def get_mse(V,v):
        e = 0
        for k in V.keys():
            e += ((V[k] - v[k])**2)
        return np.sqrt(e/len(V.keys()))

def td1(alpha=0.05,discount=1):            
    R = RWalk()    
    V = {'A':0.0,'B':0.0,'C':0.0,'D':0.0,'E':0.0}
    episode = 100
    E = []
    while episode > 0:
        S = R.start_new_episode()  
        is_episode_finish = False
        while True:
            reward,S_next,is_episode_finish = R.one_step()            
            if is_episode_finish:
                V[S] = V[S] + alpha * (reward - V[S])
                break
            else:
                V[S] = V[S] + alpha * (reward + discount * V[S_next] - V[S])                        
            S = S_next            
        E.append(get_mse(V,R.v))
        episode -= 1       
    
    return E,V
    
def td_lambda(lamda,alpha,discount=1):
    R = RWalk()
    V = {'A':0.0,'B':0.0,'C':0.0,'D':0.0,'E':0.0}
    E = []
    episode = 100
    while episode > 0:
        e = {}
        for s in V.keys():
            e[s] = 0.0
        S = R.start_new_episode()        
        while True:
            reward,S_next,is_episode_finish = R.one_step()
            if is_episode_finish:                
                error = reward - V[S]
            else:
                error = (reward + discount * V[S_next]) - V[S]
            e[S] = e[S] + 1            
            for s in V.keys():
                V[s] = V[s] + alpha * error * e[s]
                e[s] = discount * lamda * e[s]                 
            S = S_next
            if is_episode_finish:       
                break                   
        E.append(get_mse(V,R.v))            
        episode -= 1
        
    return E,V

                
        

def MC(alpha = 0.01):
    R = RWalk()    
    V = {'A':0.0,'B':0.0,'C':0.0,'D':0.0,'E':0.0}    
    E = []    
    episode = 1000
    while episode > 0:
        G = {}
        was_visited_state = set()
        S = R.start_new_episode()
        is_episode_finish = False        
        while True:
            was_visited_state.add(S)
            reward,S_next,is_episode_finish = R.one_step()
            for s in was_visited_state:
                if G.has_key(s):
                    G[s] += reward
                else:
                    G[s] = reward
            if is_episode_finish:
                break
            S = S_next
        
        for s in was_visited_state:
            V[s] = V[s] + alpha * (G[s] - V[s])
        E.append(get_mse(V,R.v))
        episode -= 1
    
    return E,V
    
    
    
    
            
            
        
        