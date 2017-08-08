#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:45:20 2017

@author: yonic
"""

import gym
import numpy as np
import math
import random
from collections import deque
import deep_learn.network as deep_net

STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_NEURONS = STATE_DIM**3
BATCH_SIZE = 50

MEMORY_CAPACITY = 100000
DISCOUNT_FACTOR = 0.99
MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
DECAY_RATE = 0.0001

class Memory:

    def __init__(self, capacity):
        self.examplers = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, sample):
        self.examplers.append(sample)

    def get_random_samples(self, num_samples):
        num_samples = min(num_samples, len(self.examplers))
        return random.sample(tuple(self.examplers), num_samples)


class Model:
    
    def __init__(self):
        #self.net = deep_net.Network([STATE_DIM,HIDDEN_NEURONS,HIDDEN_NEURONS,HIDDEN_NEURONS,ACTION_DIM])
        self.net = deep_net.Network([STATE_DIM,HIDDEN_NEURONS,ACTION_DIM])
        self.learning_rate = 0.00025
        self.explore_rate = MAX_EXPLORATION_RATE
        self.memory = Memory(MEMORY_CAPACITY)
        self.steps = 0
        self.env = gym.make('CartPole-v0')

            
        
    def act(self, s):
        if np.random.random() < self.explore_rate:
            return np.random.randint(0, ACTION_DIM)
        else:            
            return np.argmax(self.net.feedforward(s)[:,0])
            
    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

        # Reduces exploration rate linearly
        self.explore_rate = MIN_EXPLORATION_RATE + \
            (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * \
                math.exp(-DECAY_RATE * self.steps)

    def replay(self):
        
         batch = self.memory.get_random_samples(BATCH_SIZE)
         batchLen = len(batch)
         print len(self.memory.examplers)

         #states = np.array([sample[0] for sample in batch], dtype=np.float32)
         #no_state = np.zeros(STATE_DIM)
         #resultant_states = np.array([(no_state if sample[3] is None else sample[3]) for sample in batch], dtype=np.float32)     
  
         
         train = []
         for i in range(batchLen):             
             state, action, reward, resultant_state = batch[i]             
             t = np.zeros((4,1))
             t[:,0] = state[:]             
             y = self.net.feedforward(t)
             
             
             if resultant_state is not None:                 
                 t_p = np.zeros((4,1))
                 t_p[:,0] = resultant_state[:]
                 q_p = self.net.feedforward(t_p)              
                 y[action,0] = reward + DISCOUNT_FACTOR * np.amax(q_p)              
             else:
                 y[action,0] = reward           
                 
             x = t
             train.append((x,y))                        
             
             
         
         self.net.SGD(training_data=train, epochs=1, mini_batch_size=BATCH_SIZE,\
                  eta=self.learning_rate, test_data=None)
        
    
    def run_simulation(self, solved_reward_level):        
        state = self.env.reset()
        total_rewards = 0
    
        while True:
            #env.render()            
            t = np.zeros((4,1))
            t[:,0] = state
            action = self.act(t)           
    
            resultant_state, reward, done, info = self.env.step(action)
    
            if done: # terminal state
                resultant_state = None
    
            self.observe((state, action, reward, resultant_state))           
            
            self.replay()
    
            state = resultant_state
            total_rewards += reward
    
            if total_rewards > solved_reward_level or done:
                return total_rewards
                
def main():
    DONE_REWARD_LEVEL = 196
    MAX_NUM_EPISODES = 3000
    EPISODES_PER_PRINT_PROGRESS = 50
    
    episod_num = 0
    reward_sum = 0
    m = Model()
    for episod_num in xrange(MAX_NUM_EPISODES):
        reward_sum += m.run_simulation(DONE_REWARD_LEVEL*2)
        if episod_num%EPISODES_PER_PRINT_PROGRESS == 0:
            print reward_sum / EPISODES_PER_PRINT_PROGRESS,episod_num,m.explore_rate
            reward_sum = 0    
    return m
    
    

    
def test_net(net):
    env = gym.make('CartPole-v0')
    done= False
    state = env.reset()
    while not done:
        env.render()
        t = np.zeros((4,1))
        t[:,0] = state
        r = net.feedforward(t)
        action = np.argmax(r)
        resultant_state, reward, done, info = env.step(action)
    

def test__():
    net = deep_net.Network([4,64,2])
    train = []
    for i in xrange(100):
        x = np.zeros((4,1))
        y = np.zeros((2,1))
        train.append((x,y))
    
    net.SGD(train,epochs=1,mini_batch_size=10,eta=0.0025)
    
    x = np.zeros((4,1))
    print x.shape
    return net.feedforward(x)
    
        


            
    
        