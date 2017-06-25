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
class gw_sarsa:
    def __init__(self):
        self.max_state = gw.STATES[0] * gw.STATES[1]
        self.max_action = len(gw.ACTIONS)
        self.Q = pd.DataFrame(np.zeros((self.max_state,self.max_action)),columns=gw.ACTIONS)
        
        
    def sarsa(self,alpha,lamda,episod_num):
        g = gw.GridWorld()
        s = g.get_pos_flatten()
        a = 'right'
        
        while episod_num > 0:
            episod_num -= 1
            E = pd.DataFrame(np.zeros((self.max_state,self.max_action)),columns=gw.ACTIONS)
            iteration = 0
            while True:
                g.move(a)
                s_prime,r = g.get_state_and_reward()
                a_prime  = choice_gready(self.Q,s_prime)
                delta = r + self.Q.ix[s_prime,a_prime] - self.Q.ix[s,a]
                E.ix[s,a] = E.ix[s,a] + 1
                for i in self.Q.index:
                    for j in self.Q.columns:
                        self.Q.ix[i,j] = self.Q.ix[i,j] + alpha * delta * E.ix[i,j]
                        E.ix[i,j] = lamda * E.ix[i,j]
                s = s_prime
                a = a_prime
                iteration += 1
                if r == 0:
                    print "Episode %d is finishing in iteration: %d" % (episod_num,iteration)
                    break       
        
        
        
    
    