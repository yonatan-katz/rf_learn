#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:58:18 2017

@author: yonic
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from Tiles.tiles import tiles

def make_x_y():
    np.random.seed(2)
    size = int(1e4)
    x1 = np.random.normal(size=size)
    x2 = np.random.normal(size=size)
    x3 = np.random.normal(size=size)
    e = np.random.normal(size=size)
    y = 2*x1 + 3*x2 + e
    
    return pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'y':y})


def resolve_one_shot_ridge(xy,betta_ridge=0.1):    
    #train
    x = xy.ix[0:9000,['x1','x2','x3']]
    y = xy.ix[0:9000,['y']]
    xt = x.transpose()
    xtx = xt.dot(x)
    xty = xt.dot(y)    
    D = np.zeros(xtx.shape)
    np.fill_diagonal(D,np.diag(xtx))
    b = np.linalg.inv(xtx +betta_ridge*D).dot(xty)    
    
    #test
    x = xy.ix[9000:10000,['x1','x2','x3']]
    y = xy.ix[9000:10000,['y']]
    yhat = x.dot(b)
    mse = 1 - r2_score(y,yhat)    
    return b,mse
    

def resolve_sgd(xy,alpha=0.01):
    #train
    x = xy.ix[0:9000,['x1','x2','x3']]
    y = xy.ix[0:9000,['y']]
    b = np.zeros(3)    
    for i in xrange(len(x)):
        target = y.iloc[i].values
        x_row = x.iloc[i].values
        source = x_row.dot(b)
        
        b = b + alpha*(target-source)*x_row
        

    #test
    x = xy.ix[9000:10000,['x1','x2','x3']]
    y = xy.ix[9000:10000,['y']]
    yhat = x.dot(b)
    mse = 1 - r2_score(y,yhat)    
    return b,mse
    
    
def resolve_sgd_ridge(xy,alpha=0.01,betta_ridge=0.1):
    #train
    x = xy.ix[0:9000,['x1','x2','x3']]
    y = xy.ix[0:9000,['y']]
    b = np.zeros(3)    
    for i in xrange(len(x)):
        target = y.iloc[i].values
        x_row = x.iloc[i].values
        source = x_row.dot(b)
        
        #b = b + alpha* ((target-source)*x_row - betta_ridge*b)
        b = b + alpha* ((target-source)*x_row)
        

    #test
    x = xy.ix[9000:10000,['x1','x2','x3']]
    y = xy.ix[9000:10000,['y']]
    yhat = x.dot(b)
    mse = 1 - r2_score(y,yhat)    
    return b,mse    
    
def resolve_tiles_sgd(xy,alpha=0.1):
    #train
    x = xy.ix[0:9000,['x1','x2','x3']]
    y = xy.ix[0:9000,['y']]

    weights = np.zeros(512)
    #alpha = alpha / 8
    for i in xrange(len(x)):
        target = y.iloc[i]
        x_row = list(x.iloc[i])
        tiles_array = tiles(8,512,x_row)
        result = 0.0
        for i in xrange(8):
            result += weights[tiles_array[i]];
        
        for i in xrange(8):
            weights[tiles_array[i]] += alpha * (target - result);    

    #test
    x = xy.ix[9000:10000,['x1','x2','x3']]
    y = xy.ix[9000:10000,['y']]
    yhat = []
    #mse = 0.0
    for i in xrange(len(x)):
        target = y.iloc[i]
        x_row = list(x.iloc[i])
        tiles_array = tiles(8,512,x_row)
        result = 0.0
        for i in xrange(8):
            result += weights[tiles_array[i]];
        #mse += (target - result)**2
        yhat.append(result)
            
    mse = 1 - r2_score(y,yhat)
    return  mse,y,yhat
        
    
        
    
