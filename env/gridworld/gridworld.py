# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:51:11 2017

@author: yonic
"""

import numpy as np

ACTIONS = ['up','down','left','right']

STATES = (10,7)

class GridWorld:
    def __init__(self,x=STATES[0],y=STATES[1],wind_obstacle=[0,0,0,1,1,1,2,2,1,0],\
                start_width=0,start_height=3,
                end_width=7,end_height=3):
        self.x = x
        self.y = y
        self.wind_obstacle = wind_obstacle
        self.agent_pos_height = start_height
        self.agent_pos_width = start_width
        self.goal_pos_height = end_height
        self.goal_pos_width = end_width
        self.world = np.zeros((y,x))
        self.init_world()
        
    def get_pos_flatten(self):
        return self.agent_pos_height * STATES[0] + self.agent_pos_width
         
    def make_pos_agent(self):
        return (self.agent_pos_height,self.agent_pos_width)
        
    def make_pos_goal(self):
        return (self.goal_pos_height,self.goal_pos_width)
              
    def init_world(self):        
        self.world = np.zeros((self.y, self.x))
        pos_agent = self.make_pos_agent()
        self.world[pos_agent] = np.NaN
        
        pos_goal = self.make_pos_goal()
        self.world[pos_goal] = 1
        
        
    def print_one_step_solution(self,q,is_trace=False):
        if not is_trace:
            self.init_world()
            
        state,reward = self.get_state_and_reward()
        pos_agent = self.make_pos_agent()
        print pos_agent
        self.world[pos_agent] = 6
        action = np.argmax(q.ix[state,:])
        print action
        self.move(action)
        print self.world
            
        
    def get_state_and_reward(self):
        if (self.agent_pos_height == self.goal_pos_height) and \
            (self.agent_pos_width == self.goal_pos_width):
            R = 0
        else:
            R = -1
        
        state = self.get_pos_flatten()
        
        return state, R 
        
        
    def move(self,action):

        def clip():
            if self.agent_pos_height < 0:
                self.agent_pos_height= 0
            elif self.agent_pos_height >= self.y:
                self.agent_pos_height = self.y-1
                
            if self.agent_pos_width < 0:
                self.agent_pos_width = 0
            elif self.agent_pos_width >= self.x:
                self.agent_pos_width= self.x - 1
                                                        
        if action == 'up':
            self.agent_pos_height -=1
        elif action == 'down':
            self.agent_pos_height += 1
        elif action == 'left':
            self.agent_pos_width -= 1
            clip()
            self.agent_pos_height += self.wind_obstacle[self.agent_pos_width]
        elif action == 'right':
            self.agent_pos_width += 1
            clip()
            self.agent_pos_height += self.wind_obstacle[self.agent_pos_width]
        else:
            raise Exception("Not known action")
                 
        clip()
        