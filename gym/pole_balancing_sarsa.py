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

XMAX = 1
YMAX = 1
ZMAX = 6
KMAX = 3
DISCRETE_OBSERVATION_SPACE_MAX = XMAX*YMAX*ZMAX*KMAX
ACTION_NUM = 2
ACTIONS = range(ACTION_NUM)

def get_best_action(Q,observation_discrete):
    return np.argmax(Q.ix[observation_discrete,:])
    
def get_max_q_value(Q,observation_discrete):
    return np.max(Q.ix[observation_discrete,:])
    
def choice_gready(Q,observation_discrete,e=0.1):
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
    dim = [XMAX,YMAX,ZMAX,KMAX]
    i = 0
    for l,h in zip(low,high):
        space.append(np.linspace(l,h,dim[i]))
        i += 1
    return space

def convert_to_discrete_state(observation,discrete_space):
    d = []
    for i in xrange(len(observation)):
        d.append(np.digitize([observation[i]],discrete_space[i])[0] - 1)
    return d

def convert_to_flat(discrete_observation):
    x,y,z,k = discrete_observation
    return x*YMAX*ZMAX*KMAX + y*ZMAX*KMAX +z*KMAX +k
    #return x+y*XMAX+z*YMAX+k*ZMAX

def test_space():
    env = gym.make('CartPole-v0')
    o = env.observation_space
    discrete_space = make_discrete_space(o)
    print discrete_space
    s1 = convert_to_discrete_state(o.high,discrete_space)
    s2 = convert_to_discrete_state(o.low,discrete_space)
    s3 = convert_to_discrete_state((o.low+o.high)/2,discrete_space)
    print o.high,s1, convert_to_flat(s1)
    print o.low,s2, convert_to_flat(s2)
    print (o.low+o.high)/2,s3, convert_to_flat(s3)
    

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
        self.e = 0.999
 


    def save_q(self,fname):
        with open(fname,'wb') as fd:
            pickle.dump(self.Q,fd)
        
        
    def sarsa(self,alpha,lamda,episod_num,on_policy=True):
        while self.episod < episod_num:
            iteration = 0
            observation = self.env.reset()
            s_discrete = convert_to_flat(convert_to_discrete_state(observation,self.discrete_space))
            a = choice_gready(self.Q,s_discrete)
            E = pd.DataFrame(np.zeros((\
                DISCRETE_OBSERVATION_SPACE_MAX,\
                ACTION_NUM)),\
                columns=ACTIONS) 
            if self.episod % 10 ==0:
                self.e = self.e * self.e
                
            while True:
                observation_prime, r, done, info = self.env.step(a)                
                #self.env.render()
                s_prime_discrete = convert_to_flat(convert_to_discrete_state(observation_prime,self.discrete_space))
                a_prime  = choice_gready(self.Q,s_prime_discrete,self.e)
                
                if on_policy:
                    delta = r + self.Q.ix[s_prime_discrete,a_prime] - \
                        self.Q.ix[s_discrete,a]
                else:#off policy Q learning:
                    delta = r + get_max_q_value(self.Q, s_prime_discrete) - \
                        self.Q.ix[s_discrete,a]                  
                    
                E.ix[s_discrete,a] = E.ix[s_discrete,a] + 1
                for i in self.Q.index:
                    for j in self.Q.columns:
                        self.Q.ix[i,j] = self.Q.ix[i,j] + alpha * delta * E.ix[i,j]
                        E.ix[i,j] = lamda * E.ix[i,j]
                s_discrete = s_prime_discrete
                a = a_prime
                iteration += 1
                self.df.loc[self.episod] = [iteration]
                if done:
                    print "Episode %d is finishing in iteration: %d, e:%f" % (self.episod,iteration,self.e)
                    break               
               
            self.episod += 1
        return self.df
        
def full_main():
    g = mc_sarsa()
    df = g.sarsa(alpha=0.1,lamda=0.5,episod_num=200,on_policy=False)
    g.save_q(fname='q.bin')
    return g

def part_main(g,episodes=100):
    return g.sarsa(alpha=0.1,lamda=0.5,episod_num=episodes)

def test_q(Q):
    env = gym.make('CartPole-v0')
    observation = env.reset()
    discrete_space = make_discrete_space(env.observation_space)
    iterations = 0
    while True:
        env.render()
        d = convert_to_flat(convert_to_discrete_state(observation,discrete_space))
        a = get_best_action(Q,d)
        observation_prime, r, done, info = env.step(a)
        if done:
            break
        iterations += 1
        print iterations
        observation = observation_prime
    return iterations