# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:36:04 2017

@author: yonic
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 03:26:39 2017

@author: yonic
"""

import numpy as np
import pandas as pd
import cPickle as pickle
import gym

XMAX = 6
YMAX = 6
ZMAX = 6
KMAX = 6
DISCRETE_OBSERVATION_SPACE_MAX = XMAX*YMAX*ZMAX*KMAX
ACTION_NUM = 2
ACTIONS = range(ACTION_NUM)

def get_best_action(Q,observation_discrete):
    return np.argmax(Q.ix[observation_discrete,:])
    
def choice_gready(Q,observation_discrete,e=0.01):
    p = np.random.uniform()
    if p < e:
        return np.random.choice(ACTIONS,1)[0]
    else:
        return get_best_action(Q,observation_discrete)


'''Sarsa implementation for the mountain car problem
   reward is not discounted
'''

def make_discrete_space(observation):
    high = observation.high
    low = observation.low
    space = np.array([])
    space = []
    for l,h in zip(low,high):
        space.append(np.linspace(l,h,6))
    return space

def convert_to_discrete_state(observation,discrete_space):
    d = []
    for i in xrange(len(observation)):
        d.append(np.digitize([observation[i]],discrete_space[i])[0] - 1)
    return d

def convert_to_flat(discrete_observation):
    x,y,z,k = discrete_observation
    
    return x+y*XMAX+z*YMAX+k*ZMAX
    

class mc_sarsa:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.discrete_space = make_discrete_space(self.env.observation_space)
        self.Q = pd.DataFrame(np.zeros((\
                DISCRETE_OBSERVATION_SPACE_MAX,\
                ACTION_NUM)),\
                columns=ACTIONS)
        self.df = pd.DataFrame(columns=['episods'])
        self.episod = 0
        self.iteration = 0


    def save_q(self,fname):
        with open(fname,'wb') as fd:
            pickle.dump(self.Q,fd)
        
        
    def sarsa(self,alpha,lamda,episod_num):
        while self.episod < episod_num:
            observation = self.env.reset()
            s_discrete = convert_to_flat(convert_to_discrete_state(observation,self.discrete_space))
            a = choice_gready(self.Q,s_discrete)
            E = pd.DataFrame(np.zeros((\
                DISCRETE_OBSERVATION_SPACE_MAX,\
                ACTION_NUM)),\
                columns=ACTIONS)            
            while True:
                observation_prime, r, done, info = self.env.step(a)
                self.env.render()
                s_prime_discrete = convert_to_flat(convert_to_discrete_state(observation_prime,self.discrete_space))
                a_prime  = choice_gready(self.Q,s_prime_discrete)
                delta = r + self.Q.ix[s_prime_discrete,a_prime] - \
                    self.Q.ix[s_discrete,a]
                E.ix[s_discrete,a] = E.ix[s_discrete,a] + 1
                for i in self.Q.index:
                    for j in self.Q.columns:
                        self.Q.ix[i,j] = self.Q.ix[i,j] + alpha * delta * E.ix[i,j]
                        E.ix[i,j] = lamda * E.ix[i,j]
                s_discrete = s_prime_discrete
                a = a_prime
                self.iteration += 1
                self.df.loc[self.iteration] = [self.episod]
                if done:
                    print "Episode %d is finishing in iteration: %d" % (self.episod,self.iteration)
                    break
               
            self.episod += 1
        return self.df
        
def main():
    g = mc_sarsa()
    df = g.sarsa(alpha=0.1,lamda=0.5,episod_num=1)
    #g.save_q(fname='C:\\Users\\yonic\\projects\\mountain_car\\q.bin')
    return g
