# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 03:26:39 2017

@author: yonic
"""

import numpy as np
import pandas as pd
import cPickle as pickle
import gym

ACTION_NUM = 3
VEL_LOW = -0.7
VEL_HIGH = 0.7
POS_LOW = -1.2
POS_HIGH = 0.6
ACTIONS = range(ACTION_NUM)

def get_best_action(Q,observation_discrete):
    return np.argmax(Q.ix[observation_discrete,:])
    
def choice_gready(Q,observation_discrete,e=0.3):
    p = np.random.uniform()
    if p < e:
        return np.random.choice(ACTIONS,1)[0]
    else:
        return get_best_action(Q,observation_discrete)


'''Sarsa implementation for the mountain car problem
   reward is not discounted
'''


class mc_sarsa:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.num_of_bins = 10
        self.pos_space = np.linspace(POS_LOW,POS_HIGH,self.num_of_bins)
        self.vel_space = np.linspace(VEL_LOW,VEL_HIGH,self.num_of_bins)
        self.num_of_pos_bin = len(self.pos_space) + 1
        self.num_of_vel_bin = len(self.vel_space) + 1
        self.Q = pd.DataFrame(np.zeros((\
                self.num_of_pos_bin *  self.num_of_vel_bin,\
                ACTION_NUM)),\
                columns=ACTIONS)
        self.df = pd.DataFrame(columns=['episods'])
        self.episod = 0
        self.iteration = 0
    '''Bining and flaten observation state
    '''
    def convert_to_discrete_state(self,observation):
        pos,vel = observation
        pos_d = np.digitize([pos],self.pos_space)[0]-1
        pos_vel = np.digitize([vel],self.vel_space)[0]-1
        
        return pos_d + pos_vel * self.num_of_pos_bin
        
        
     
    def save_q(self,fname):
        with open(fname,'wb') as fd:
            pickle.dump(self.Q,fd)
        
        
    def sarsa(self,alpha,lamda,episod_num):
        while self.episod < episod_num:
            observation = self.env.reset()
            s_discrete = self.convert_to_discrete_state(observation)
            a = choice_gready(self.Q,s_discrete)
            E = pd.DataFrame(np.zeros((\
                self.num_of_pos_bin *  self.num_of_vel_bin,\
                ACTION_NUM)),\
                columns=ACTIONS)            
            while True:
                observation_prime, r, done, info = self.env.step(a)
                s_prime_discrete = self.convert_to_discrete_state(observation_prime)
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
                if r == 0:
                    print "Episode %d is finishing in iteration: %d" % (self.episod,self.iteration)
                    break
               
            self.episod += 1
        return self.df
        
def main():
    g = mc_sarsa()
    df = g.sarsa(alpha=0.1,lamda=0.5,episod_num=1)
    g.save_q(fname='C:\\Users\\yonic\\projects\\mountain_car\\q.bin')
    return g
