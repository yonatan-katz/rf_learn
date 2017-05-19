#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:18:17 2017

@author: yonic
"""
import gym
import numpy as np
import pandas as pd

def get_env_distr(env): 
    if env.observation_space.__class__ == gym.spaces.box.Box:
        box_shape = env.observation_space.shape[0]
        s = env.observation_space.sample()
        for i in xrange(10000):
            s = np.concatenate([s,env.observation_space.sample()])
        print i,box_shape
        return pd.DataFrame(s.reshape((i+2,box_shape)))
        
    else:
        raise Exception('Not supported observation class')