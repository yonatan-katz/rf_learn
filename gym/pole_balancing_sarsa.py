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


NUMBER_OF_BINS = 6
NUMBER_OF_SPACE_DIMENSIONS = 4
NUMBER_OF_ACTIONS = 2

def get_best_action(Q,observation_discrete):
    return np.argmax(Q.ix[observation_discrete,:])
    
def choice_gready(Q,observation_discrete,e=0.3):
    p = np.random.uniform()
    if p < e:
        return np.random.choice(range(NUMBER_OF_ACTIONS),1)[0]
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
        space.append(np.linspace(l,h,NUMBER_OF_BINS))
    return space

def convert_to_discrete_state(observation,discrete_space):
    d = []
    for i in xrange(len(observation)):
        d.append(np.digitize([observation[i]],discrete_space[i])[0] - 1)
    return d

def convert_to_flat_state(discrete_state):
    x = discrete_state[0]
    y = discrete_state[1]
    z = discrete_state[2]
    h = discrete_state[3]

    return (h*NUMBER_OF_BINS*3 + z*NUMBER_OF_BINS*2 + y*NUMBER_OF_BINS*1 + x)
    
def test():
    env = gym.make('CartPole-v0')        
    discrete_space = make_discrete_space(env.observation_space)    
    while True:
        observation = env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print reward,done
            discrete_observation = convert_to_discrete_state(observation,discrete_space)
            flaten = convert_to_flat_state(discrete_observation)
            #print flaten, discrete_observation,reward,done
            if done:
                break
            
        
    
    
    

class pb_sarsa:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.discerte_space = make_discrete_space(self.env.observation_space)        
        self.Q = pd.DataFrame(np.zeros((\
                NUMBER_OF_BINS**NUMBER_OF_SPACE_DIMENSIONS,NUMBER_OF_ACTIONS)),\
                columns=range(NUMBER_OF_ACTIONS))
        self.df = pd.DataFrame(columns=['episods'])
        self.episod = 0
        self.iteration = 0       
        
     
    def save_q(self,fname):
        with open(fname,'wb') as fd:
            pickle.dump(self.Q,fd)
        
        
    def sarsa(self,alpha,lamda,iter_num):
        while self.iteration < iter_num:
            observation = self.env.reset()
            discrete_observation = convert_to_discrete_state(observation,self.discerte_space)
            s_flaten = convert_to_flat_state(discrete_observation)
            a = choice_gready(self.Q,s_flaten)
            E = pd.DataFrame(np.zeros((\
                NUMBER_OF_BINS**NUMBER_OF_SPACE_DIMENSIONS,NUMBER_OF_ACTIONS)),\
                columns=range(NUMBER_OF_ACTIONS))    
            while True:
                observation_prime, r, done, info = self.env.step(a)
                if done:
                    r = 0
                discrete_observation = convert_to_discrete_state(observation_prime,self.discerte_space)
                s_flaten_prime = convert_to_flat_state(discrete_observation)          
                a_prime  = choice_gready(self.Q,s_flaten_prime)
                delta = r + self.Q.ix[s_flaten_prime,a_prime] - \
                    self.Q.ix[s_flaten,a]
                E.ix[s_flaten,a] = E.ix[s_flaten,a] + 1
                for i in self.Q.index:
                    for j in self.Q.columns:
                        self.Q.ix[i,j] = self.Q.ix[i,j] + alpha * delta * E.ix[i,j]
                        E.ix[i,j] = lamda * E.ix[i,j]
                s_flaten = s_flaten_prime
                a = a_prime
                self.iteration += 1
                self.df.loc[self.iteration] = [self.episod]
                if done:                    
                    print "Episode %d is finishing in iteration: %d" % (self.episod,self.iteration)
                    break
               
            self.episod += 1
        return self.df
        
def main():
    g = pb_sarsa()
    df = g.sarsa(alpha=0.1,lamda=0.5,iter_num=1000)
    g.save_q(fname='/home/yonic/projects/pool_balancing/q.bin')
    return g
