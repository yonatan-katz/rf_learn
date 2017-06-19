# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:51:11 2017

@author: yonic
"""
import numpy as np

class GridWorld:
    def __init__(self,x=10,y=7,wind_obstacle=[0,0,0,1,1,1,2,2,1,0],\
                start_width=0,start_height=3,
                end_width=7,end_height=3):
        self.x = x
        self.y = y
        self.wind_obstacle=wind_obstacle
        self.agent_pos_height = start_height
        self.agent_pos_width = start_width
        self.goal_pos_height = end_height
        self.goal_pos_width = end_width
        self.world = np.zeros((y,x))
        
    def make_pos_agent(self):
        return (self.agent_pos_height,self.agent_pos_width)
        
    def make_pos_goal(self):
        return (self.goal_pos_height,self.goal_pos_width)
              
    def print_world(self):        
        world = np.zeros((self.y, self.x))
        pos_agent = self.make_pos_agent()
        world[pos_agent] = np.NaN
        
        pos_goal = self.make_pos_goal()
        world[pos_goal] = 1
        print world
        
    def get_state(self):
        if self.agent_pos == self.end:
            R = 1
        else:
            R = 0
        
        return self.agent_pos,R 
        
        
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
            print self.agent_pos_height
        elif action == 'down':
            self.agent_pos_height += 1
        elif action == 'left':
            self.agent_pos_width -= 1
            self.agent_pos_height += self.wind_obstacle[self.agent_pos_width]
        elif action == 'right':
            self.agent_pos_width += 1
            self.agent_pos_height += self.wind_obstacle[self.agent_pos_width]
        else:
            raise Exception("Not known action")
        
        clip()
        