'''
Created on 2018. 7. 8.

@author: 김기현
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from math import exp
from six.moves import input
import pandas as pd
import itertools


S = input('Stock price : ')
r = input('Interest rate(%) : ')
K = input('Strike price : ')
T = input('Period : ')

filename = 'TWTRf.csv'
myframe = pd.read_csv(filename, encoding ='utf-8')
#print(myframe)
mygrouping = myframe['Adj Close']
Sigma = mygrouping.std()
print('Standard Deviation : ',  Sigma)
#n_defaults = np.random.binomial(n=100,p=0.02,size=10000)
def European_Put_Option(S,T, r, K,Sigma):
    S = int(S)
    T=int(T)
    K=int(K)
    u = exp((Sigma/100)*math.sqrt(T))
    d = exp(-(Sigma/100)*math.sqrt(T))
    
    r = float(r)/100
    q1 = ((1+int(r))-d)/(u-d)
    q2 = 1 - q1 
   
    Su =0
    for k in range(0,T+1):
        E = K-(S*((u)**k)*((d)**(T-k)))
        
        if E > 0:
           
           fact =  math.factorial(T)/(math.factorial(T-k)*math.factorial(k))
           Q = (q1**k)*(q2**(T-k))
           Su = fact*E*Q + Su
        else :
            Su = Su + 0
        P0 = (1/(1+r)**T)*Su
    print('the price of Put option(P0) : ',P0)

An = European_Put_Option(S, T, r, K, Sigma)