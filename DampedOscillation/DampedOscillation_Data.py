import numpy as np
import csv
import pandas as pd 


end = 10
step = 30

a = 1
w = 2
b = 0.2

w1=w

w2=(w1**2-b**2)**0.5

error1 = 0.01

def Data(x):
    return a * np.exp(-b*x)*np.cos(w2*x)

x = np.linspace(0,end,step)
y = Data(x)

z = []

for i in range(step):
    z.append(Data(x[i]+ np.mean(np.random.randn(1)*error1)))
     
file = open('DampedOscillation/Data.csv','w',newline = '')

writer = csv.writer(file)

writer.writerow(['x','y','z']) 

for i in range(0,len(x)):
    writer.writerow([x[i],y[i],z[i]]) 

import matplotlib.pyplot as plt

plt.plot(x,y,x,z)
plt.show()

file.close


