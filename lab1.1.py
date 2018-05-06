# -*- coding:utf-8 -*-
import  os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl

a0 = []
a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []
a7 = []
a8 = []
a9 = []
re=[]
n = 0
filename = 'magic04.txt'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        n += 1
        if not lines:
            break
        tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,temp = [i for i in lines.split(",")]
        a0.append(float(tmp0))
        a1.append(float(tmp1))
        a2.append(float(tmp2))
        a3.append(float(tmp3))
        a4.append(float(tmp4))
        a5.append(float(tmp5))
        a6.append(float(tmp6))
        a7.append(float(tmp7))
        a8.append(float(tmp8))
        a9.append(float(tmp9))
re.append(sum(a0)/n)
re.append(sum(a1)/n)
re.append(sum(a2)/n)
re.append(sum(a3)/n)
re.append(sum(a4)/n)
re.append(sum(a5)/n)
re.append(sum(a6)/n)
re.append(sum(a7)/n)
re.append(sum(a8)/n)
re.append(sum(a9)/n)

#多元均值向量
print re

#协方差
x=np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9])
print(np.cov(x))

#相关性、散点图
def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)
s0=np.array(a0)
s1=np.array(a1)
print cos(s0,s1)  #相关性
plt.plot(s0,s1,'k.')
plt.show()

#概率分布图
from scipy import stats
from scipy.stats import norm
X = norm(loc=1.0,scale=2.0)
pl.plot(s0,X.pdf(s0),label="$X$",color="red")
plt.show()

def variance(a):
    array = np.array(a)
    var = array.var()
    return var
t=[]
t.append(variance(a0))
t.append(variance(a1))
t.append(variance(a2))
t.append(variance(a3))
t.append(variance(a4))
t.append(variance(a5))
t.append(variance(a6))
t.append(variance(a7))
t.append(variance(a8))
t.append(variance(a9))
min=t[0]
max=min
n=0
x=0
for i in range(1,10):
    if t[i]<min:
        min=t[i]
        n=i
    if t[i]>max:
        max = t[i]
        x=i
print ("%d %s"%(n,min))
print ("%d %s"%(x,max))

def cov(a):
    array = np.array(a)
    cov = np.cov(array)
    return cov
m=[]
m.append(cov(a0))
m.append( cov(a1))
m.append( cov(a2))
m.append( cov(a3))
m.append( cov(a4))
m.append( cov(a5))
m.append( cov(a6))
m.append( cov(a7))
m.append( cov(a8))
m.append( cov(a9))
min1=m[0]
max1=min1
n=0
x=0
for i in range(1,10):
    if m[i]<min1:
        min1=m[i]
        n=i
    if m[i]>max1:
        max1 = m[i]
        x=i
print ("%d %s"%(n,min1))
print ("%d %s"%(x,max1))