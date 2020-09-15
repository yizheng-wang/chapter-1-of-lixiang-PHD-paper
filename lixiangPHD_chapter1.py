#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:29:50 2020

@author: wangyizheng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv,det

imgtot_training = 1
imgtot = 1
imgtot_testing = 1 
cat_training = 1
lrnum = 0.0001
epochnum = 200000


class Mesh:
    def __init__(self, ex, ey, lx, ly):
        self.ex, self.ey = ex,ey
        self.lx, self.ly = lx,ly
        self.nx, self.ny = ex+1, ey+1
        self.hx, self.hy = lx/ex, ly/ey
        # initial the node coordinates
        self.nodes = []
        for y in np.linspace(0, ly, self.ny):
            for x in np.linspace(0, lx, self.nx):
                self.nodes.append([x, y])
        self.nodes = np.array(self.nodes) # change the list type of self.nodes to array
        self.conn = [] # element coordinate
        for j in range(self.ey):
            for i in range(self.ex):
                n0 = i + j*self.nx
                self.conn.append([n0, n0+1, n0+1+self.nx, n0+self.nx])
    def num_nodes(self):return self.nx*self.ny
    def num_elements(self):return self.ex*self.ey
    
def shape(xi):
    x, y = tuple(xi)
    N = [(1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)]
    return 0.25*np.array(N)
def gradshape(xi):
    x,y = tuple(xi)
    dN = [[-(1-y), (1-y), (1+y), -(1+y)], 
          [-(1-x), -(1+x), (1+x), (1-x)]] 
    return np.array(dN)

print('constructing mesh')
mesh = Mesh(20, 20, 2, 2 )
load = np.zeros(2*mesh.num_nodes())

E,v = 100, 0.3
C = E/(1+v)/(1-2*v)*np.array([[1-v, v, 0],
                             [v, 1-v, 0],
                             [0, 0, 0.5-v]])
K = np.zeros((mesh.num_nodes()*2, mesh.num_nodes()*2))
q4 = [[x/np.sqrt(3), y/np.sqrt(3)] for y in [-1, 1] for x in [-1, 1]] # q4 is gauss integrate position
print('assembling the stiffness matrix')
B = np.zeros((3, 8))
for c in mesh.conn:
    xIe = mesh.nodes[c, :]
    Ke = np.zeros((8, 8))
    for q in q4:
        dN = gradshape(q) # get the physic coordinate dN
        J = np.dot(dN, xIe)
        dN = np.dot(inv(J), dN)
        B[0, 0::2] = dN[0, :]
        B[1, 1::2] = dN[1, :]
        B[2, 0::2] = dN[1, :]
        B[2, 1::2] = dN[0, :]
    
        Ke += np.dot(np.dot(B.T, C), B) * det(J)
        
    for i,I in enumerate(c):
        for j,J in enumerate(c):
            K[2*I, 2*J] += Ke[2*i, 2*j]
            K[2*I+1, 2*J] += Ke[2*i+1, 2*j]
            K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]
            K[2*I, 2*J+1] += Ke[2*i, 2*j+1]
            
f = np.zeros((2*mesh.num_nodes()))
for i in range(mesh.num_nodes()):
   if mesh.nodes[i, 1] == 0:
        K[2*i, :] = 0
        K[2*i+1, :] = 0
        K[2*i, 2*i] = 1
        K[2*i+1, 2*i+1] = 1
   if mesh.nodes[i, 1] == mesh.ly:
        x = mesh.nodes[i, 0]
        fbar = 10000
        f[2*i+1] = fbar
print('solving linear system')
load = f
print(f)
print(K)
imgsize_x = K.shape[0]
imgsize_y = K.shape[1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, imgsize_x-42), bias = False) # stide = 0, padding = 0 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # because input has 4 dimension, batch, channel, x, y respectively, so that return the last 3 dimension and flat it
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
net.cuda()
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = lrnum)
target = torch.from_numpy(load)[None, 42:].float().cuda() # make the target shape [1, 882], because output shape is [1, 882]
input = (torch.from_numpy(K))[None, None, 42:, 42:].float().cuda() # make the input shape [1, 1, 882, 882], because input shape must have 4 dimensions.
print(target)
filename = 'loss.dat'
with open(filename, 'a') as f:
    f.write('loss\n')
    for epoch in range(epochnum):
        running_loss = 0
        loss_avg = 0
        for k in range(imgtot_training):
            
            output = net(input)
            '''lambda_p = 1e8
            punish = lambda_p*net.conv1.weight[:, :, :, :42].sum().item()'''
            '''+torch.tensor(punish)'''
            loss = criterion(output, target).float() 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss
            if epoch % 100 == 0 : # print every 100 epochs
                print(f'epochs : {epoch}, and loss : {running_loss}')
            f.write(str(running_loss) + '\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    