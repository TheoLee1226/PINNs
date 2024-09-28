import torch

import torch.autograd as autograd

import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np

import time

import pandas as pd 

from tqdm import trange

torch.set_default_dtype(torch.float)

torch.manual_seed(1234)
np.random.seed(1234)

torch.device('cuda')
print("Device:" + torch.cuda.get_device_name())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

steps=5000
lr=1e-3 #learning rate
layers = np.array([1,50,100,100,200,100,100,50,1])

x_0 = 1
omega = 2
bata = 0.2

#bounding conduction BC(0)=x_0
def f_BC(x):
    return torch.ones_like(x).to(device)*x_0

def PDE(x):
    return torch.zeros_like(x).to(device)

class FCN(nn.Module):

    def __init__(self,layers):
        super().__init__()

        self.activation = nn.Tanh()

        self.loss_function = nn.MSELoss(reduction='mean') 
 
        self.linears = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)  

    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossBC(self,x_BC):
        loss_BC = self.loss_function(self.forward(x_BC),f_BC(x_BC))
        return loss_BC
    
    def lossPDE(self,x_PDE):
        g = x_PDE.clone()
        g.requires_grad = True
        f = self.forward(g)

        f_x = autograd.grad(f,g,torch.ones_like(f).to(device),retain_graph=True, create_graph=True)[0]
        f_xx = autograd.grad(f_x,g,torch.ones_like(f).to(device),retain_graph=True, create_graph=True)[0]
        f_PDE = f_xx + 2*bata*f_x + omega**2*f
        loss_PDE=self.loss_function(f_PDE,PDE(g))
        
        return loss_PDE
    
    def loss(self,x_BC,x_PDE):
        loss_bc=self.lossBC(x_BC)
        loss_pde=self.lossPDE(x_PDE)
        return loss_bc+loss_pde

Data = open('DampedOscillation/Data.csv','r',newline = '')
Datareader = pd.read_csv(Data)
x_Data = torch.Tensor(Datareader['x'])
y_Data = torch.Tensor(Datareader['y'])
z_Data = torch.Tensor(Datareader['z'])

x=x_Data.view(-1,1)
y=y_Data.view(-1,1)
z=z_Data.view(-1,1)

BC = torch.Tensor([0]).view(1,-1)

all_train = BC

idx = 0

x_BC = BC

x_PDE = x
y_PDE = y
z_PDE = z

torch.manual_seed(1234)

x_PDE=x_PDE.float().to(device)
x_BC=x_BC.to(device)

model = FCN(layers)

print(model)

model.to(device)

params = list(model.parameters())
optimizer = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)
start_time = time.time()

losslist = []

for i in trange(steps):
    yh = model(x_BC)
    loss = model.loss(x_BC,x_PDE)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losslist.append(loss.item())
    
xh = torch.linspace(0,20,1000).view(-1,1)
yh=model(xh.to(device))
#Error
#print(model.lossBC(x.to(device)))

bata_Real = bata
omega_Real = omega
x_Real = np.linspace(0,20,1000)

def Data(x):
    return 1 * np.exp(-bata_Real*x)*np.cos((omega_Real**2-bata_Real**2)**0.5*x)

#z_plot=z.detach().cpu().numpy()
y_plot=y.detach().cpu().numpy()
yh_plot=yh.detach().cpu().numpy()

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.scatter(x,y,color='green',label='Data')
ax1.plot(xh,yh_plot,color='red',label='PINNs')
ax1.plot(x_Real,Data(x_Real),color='blue',label='Real')
ax1.set_xlabel('x',color='black')
ax1.set_ylabel('f(x)',color='black')
#ax1.tick_params(axis='y', color='black')
ax1.legend(loc = 'upper left')

ax2.plot(np.linspace(0,len(losslist),len(losslist)),losslist)
ax2.set_xlabel("Iterate")
ax2.set_ylabel("Loss")
ax2.set_ylim(0,1)

plt.show()