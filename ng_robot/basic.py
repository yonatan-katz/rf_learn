#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:49:53 2016

@author: yonic
"""
import numpy as np

def make_reward():
    R=np.matrix([-0.002]*12).reshape(3,4)
    R[0,3]=1
    R[1,3]=-1
    return R
    
#R = make_reward()

'''Actions:
    UP,DOWN,LEFT,RIGHT
'''
R = {
     1 : {1:-0.02, 2:-0.02, 3:-0.02},
     2 : {1:-0.02, 2:-0.02, 3:-0.02},
     3 : {1:-0.02, 2:-0.02, 3: -0.02},
     4 : {1:-0.02, 2:-1, 3: 1},
}


ACTIONS = ['UP','DOWN','LEFT','RIGHT']

P = {
     1 : {1:{'UP':[(1,2,0.8),(2,1,0.1),(1,1,0.1)],'DOWN':[],'LEFT':[],'RIGHT':[(2,1,0.8),(1,2,0.1),(1,1,0.1)]},
          2:{'UP' :[(1,3,0.8),(1,2,0.2)],'DOWN':[(1,1,0.8),(1,2,0.2)],'LEFT':[(1,2,0.8),(1,1,0.1),(1,3,0.1)],'RIGHT':[(1,2,0.8),(1,1,0.1),(1,3,0.1)]},
          3:{'UP':[],'DOWN':[(1,2,0.8),(2,3,0.1), (1,3,0.1)],'LEFT':[],'RIGHT':[(2,3,0.8),(1,2,0.1),(1,3,0.1)]}},
     
     2 : {1:{'UP':[(1,1,0.1),(3,1,0.1), (2,1,0.8)],'DOWN':[],'LEFT':[(1,1,0.8),(2,1,0.2)],'RIGHT':[(3,1,0.8),(2,1,0.2)]},
          2:{'UP':[],'DOWN':[],'LEFT':[],'RIGHT':[]},
          3:{'UP':[],'DOWN':[],'LEFT':[(1,3,0.8),(2,3,0.2)],'RIGHT':[(3,3,0.8),(2,3,0.2)]}},
     
     3 : {1:{'UP':[(3,2,0.8),(2,1,0.1),(4,1,0.1)],'DOWN':[],'LEFT':[(2,1,0.8), (3,2,0.1),(3,1,0.1)],'RIGHT':[(4,1,0.8), (3,2,0.1), (3,1,0.1)]},
          2:{'UP':[(3,3,0.8),(4,2,0.1),(3,2,0.1)],'DOWN':[(3,1,0.8),(4,2,0.1),(3,2,0.1)],'LEFT':[],'RIGHT':[(4,2,0.8),(3,3,0.1),(3,1,0.1)]},
          3:{'UP':[],'DOWN':[(3,2,0.8),(2,3,0.1),(4,3,0.1)],'LEFT':[(2,3,0.8),(3,2,0.1),(3,3,0.1)],'RIGHT':[(4,3,0.8),(3,2,0.1),(3,3,0.1)]}},
     
     4:  {1:{'UP':[(4,2,0.8),(3,1,0.1),(4,1,0.1)],'DOWN':[],'LEFT':[(3,1,0.8), (4,2,0.1),(4,1,0.1)],'RIGHT':[]},
          2:{'UP':[(4,3,0.8), (3,2,0.1),(4,2,0.1)],'DOWN':[(4,1,0.8),(3,2,0.1),(4,2,0.1)],'LEFT':[(3,2,0.8),(4,1,0.1),(4,3,0.1)],'RIGHT':[]},
          3:{'UP':[],'DOWN':[(4,2,0.8),(3,3,0.1),(4,3,0.1)],'LEFT':[(3,3,0.8),(4,2,0.1),(4,3,0.1)],'RIGHT':[]}},
}


def get_trp(i,j,action):
    tr  =[]
    
    def is_stuck_state(i,j):                
        if (i==2 and j==2):
            return True
        else:
            return False
            
    def is_valid_state(i,j):
        if (j<=3) and (j>=1) and (i>=1) and (i<=4):
            return True
        else:
            return False
        
    if action == 'UP':        
        j1 = j+1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.8))            
        else:
            tr.append((i,j,0.8))
        
        i1 = i-1
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.1))                 
        else:
            tr.append((i,j,0.1))                
            
        i1 = i+1
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.1))
        else:
            tr.append((i,j,0.1))
            
    elif action == 'LEFT':
        i1= i-1
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.8))
        else:
            tr.append((i,j,0.8))                
            
        j1 = j+1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.1))
        else:
            tr.append((i,j,0.1))
            
        j1 = j-1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.1))
        else:
            tr.append((i,j,0.1))
            
    elif action == 'RIGHT':                    
        i1 = i+1        
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.8))            
        else:
            tr.append((i,j,0.8))                
            
        j1 = j+1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.1))
        else:
            tr.append((i,j,0.1))
            
        j1 = j-1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.1))
        else:
            tr.append((i,j,0.1))
            
    elif action == 'DOWN':
        j1 = j-1
        if is_valid_state(i,j1) and not is_stuck_state(i,j1):
            tr.append((i,j1,0.8))            
        else:
            tr.append((i,j,0.8))
        
        i1 = i-1
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.1))                 
        else:
            tr.append((i,j,0.1))                
            
        i1 = i+1
        if is_valid_state(i1,j) and not is_stuck_state(i1,j):        
            tr.append((i1,j,0.1))
        else:
            tr.append((i,j,0.1))
            
    return tr
            
    
POLICY = {
         1:{1:'RIGHT',2:'DOWN', 3:'RIGHT'},
         2:{1:'RIGHT',2: 'NONE',3:'RIGHT'},
         3:{1:'UP', 2: 'RIGHT', 3: 'RIGHT'},
         4:{1:'UP', 2: 'NONE', 3:'NONE'}
}

def make_states():
    s = []
    for i in [1,2,3,4]:
        for j in [1,2,3]:
            s.append((i,j))
    return s   
    
STATE_SHAPE = (4,3)


'''Get probability transistion matrix, and reward function for each state
'''
def matrix_get_v():        
    S = make_states()
    M = np.zeros((len(S),len(S)))
    r = np.zeros(len(S)) 
    index = 0
    for i1,j1 in S:        
        #print (i1,j1)
        a = POLICY[i1][j1]
        TR = []        
        if a!='NONE':            
            for i2,j2,p in get_trp(i1,j1,a):
               TR.append((i2,j2,p))
               M[index][(i2-1)*3+(j2-1)] += p
        
        print (i1,j1,a)," ->", TR
        r[index] = R[i1][j1]
        index +=1
    
    return M,r
    
'''Return value function for the each state
'''
def solve(M,r):
    return np.linalg.inv((np.identity(len(M)) - 0.99*M)).dot(r)
    
def get_util_for_policy(policy=POLICY):
    m,r = matrix_get_v()
    
    v = solve(m,r)
    s = make_states()
    r = []
    
    for i in range(len(s)):
        print s[i], v[i]
        r.append((s[i], v[i]))
        
    return r
    

    
    
def value_iteration(states=make_states()): 
    diff = []
    V = np.zeros((4,3))
    V[4-1][3-1] = 1.0
    V[4-1][2-1] = -1.0

    
    #while True:
    for k in range (40):
        v1 = V.flatten()
        for i in range(V.shape[0]):
            for j in range (V.shape[1]):
                if (j==2 and i==3) or (j==1 and i==3) or (j==1 and i==1):
                    continue
                max_v = -100000.0
                for a in ACTIONS:
                    s = get_trp(i+1,j+1,a)                    
                    t = 0
                    for i1,j1,p in s:
                        t += (V[i1-1][j1-1] * p)
                    
                    t = t* 0.99                    
                    if t > max_v:
                        max_v = t                
                
                V[i][j] = R[i+1][j+1] + max_v
        print V
        
        v2 = V.flatten()
        diff.append(np.linalg.norm(v1-v2))
    
    #T = V.transpose()
    #Vt = np.copy(T)
    #Vt[0,:] = T[2,:]
    #Vt[2,:] = T[0,:]
        
    
    return diff,V
    
    
def get_best_policy(V_star):
    P_star = np.zeros((4,3),dtype=np.object)
    for i in range(P_star.shape[0]):
        for j in range(P_star.shape[1]):
            if (j==2 and i==3) or (j==1 and i==3) or (j==1 and i==1):
                    continue
            max_v = -100000.0
            best_action = 'NONE'
            for a in ACTIONS:
                s = get_trp(i+1,j+1,a)
                print s
                t = 0
                for i1,j1,p in s:
                    print i1,j1
                    t += (V_star[i1-1][j1-1] * p)
                t = t* 0.99                    
                if t > max_v:
                    max_v = t
                    best_action = a
            
            P_star[i][j] = best_action

    T = P_star.transpose()
    Pt = np.copy(T)
    Pt[0,:] = T[2,:]
    Pt[2,:] = T[0,:]
    return Pt
        
             
                
            
            
    
    
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    