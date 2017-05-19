#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:48:02 2017

@author: yonic
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import gym

def test():
    env = gym.make('MountainCar-v0')
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))                
                break
            
            
def get_state_features(s):
        s1 = s[0]
        s2 = s[1]
        interaction = s1*s2
        s1_pow_2 = np.power(s1,2)
        s2_pow_2 = np.power(s2,2)
        interaction_pow_2 = np.power(interaction,2)
        s1_pow_3 = np.power(s1,3)
        s2_pow_3 = np.power(s2,3)
        interaction_pow_3 = np.power(interaction,3)
        
        return np.array([1,s1,s2,interaction,\
                         s1_pow_2,s2_pow_2,interaction_pow_2,\
                         s1_pow_3,s2_pow_3,interaction_pow_3])
         
LOW = np.array([-1.2 , -0.07])
HIGH = np.array([ 0.6 ,  0.07])

def plot_v_func(weights):       
    x  = []
    y = []
    z = []
    for i in np.linspace(LOW[0],HIGH[0],50):
       for j in  np.linspace(LOW[1],HIGH[1],50):
          x.append(i) 
          y.append(j)
          z.append(np.abs(get_state_features(np.array([i,j])).dot(weights)))         
    
    
    # Plot the surface
    # Plot the surface    
    plt.gcf().clear()
    fig = plt.figure(1)    
    ax = fig.gca(projection='3d')
    
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)     
    plt.show()
            
            
def solve():            
    env = gym.make('MountainCar-v0')    
    betta = np.zeros(10)
    actions = [0,1,2]
    DISCOUNT = 0.9
    LEARNING_RATE = 0.1
    RIDGE_LAMBDA = 0.01
    GREEDY_ACTION = 0.5
    for e in xrange(1000):
        print betta,e
        done = False
        s = env.reset()
        plot_v_func(betta)        
        while not done:
            v_e = []
            state = env.env.state
            x = get_state_features(s)
            next_state_cache = {}
            for a in actions:
               s_tag,r,done,info = env.step(a)
               env.env.state = state
               if done:
                   break
               x_tag = get_state_features(s_tag)
               next_state_cache[a] = [r,x_tag]
               v = r + DISCOUNT * x_tag.dot(betta)
               v_e.append(v)
            
            if not done:
                #v_target = np.max(v_e)
                action_target = np.argmax(v_e)
                if np.random.normal() < GREEDY_ACTION:
                    action_target = np.random.choice(actions)
                print action_target
                betta = betta + LEARNING_RATE * (next_state_cache[a][0]*x  - x*(x - DISCOUNT*next_state_cache[a][1]).dot(betta))
                s,r,done,info = env.step(action_target)
                #betta = betta + LEARNING_RATE * (x * (v_target - x.dot(betta)) - RIDGE_LAMBDA*betta)
                #betta = betta + LEARNING_RATE * (x * (v_target - x.dot(betta)))
                env.render()
        
        
        
         
           
           
            
        
        
