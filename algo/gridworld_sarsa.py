# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:18:13 2017

@author: yonic
"""

import numpy as np
import pandas as pd
import rf_learn.env.gridworld.gridworld as gw

def get_best_action(Q,s):
    return np.argmax(Q.ix[s,:])
    
def choice_gready(Q,s,e=0.1):
    p = np.random.uniform()
    if p < e:
        return np.random.choice(gw.ACTIONS,1)[0]
    else:
        return get_best_action(Q,s)
    
'''Sarsa implementation for the ,windy grid world problem
   reward is not discounted
'''
def sarsa(alpha,lamda,episod_num):
    max_state = gw.STATES[0] * gw.STATES[1]
    max_action = len(gw.ACTIONS)
    Q = np.zeros((max_state,max_action))
    Q = pd.DataFrame(Q,columns=gw.ACTIONS)
    
    g = gw.GridWorld()
    s = g.get_pos_flatten()
    a = 'right'
    
    while episod_num > 0:
        episod_num -= 1
        E = pd.DataFrame(np.zeros((max_state,max_action)),columns=gw.ACTIONS)
        while True:
            g.move(a)
            s_prime,r = g.get_state_and_reward()
            a_prime  = choice_gready(Q,s_prime)
            delta = r + Q.ix[s_prime,a_prime] - Q.ix[s,a]
            E.ix[s,a] = E.ix[s,a] + 1
            for i in Q.index:
                for j in Q.columns:
                    Q.ix[i,j] = Q.ix[i,j] + alpha * delta * E.ix[i,j]
                    E.ix[i,j] = lamda * E.ix[i,j]
            s = s_prime
            a = a_prime
            if r == 0:
                break       
        
        
        
    
    