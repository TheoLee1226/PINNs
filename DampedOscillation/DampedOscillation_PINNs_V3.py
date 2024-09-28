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

steps=50000
lr=1e-4 #learning rate
layers = np.array([1,50,100,200,100,50,3])

#for output {y,2bata,omega^2}

omega = 2
bata = 0.2
x_0 = 1
v_0 = -bata

#bounding conduction
def f_BC_1(self,x):
    f = self.forward(x)
    return f[:,0]

#bounding conduction
def f_BC_2(self,x):
    g = x.clone()
    g.requires_grad = True
    f = self.forward(g)
    f_x = autograd.grad(f,g,torch.ones_like(f).to(device),retain_graph=True, create_graph=True)[0]
    return f_x[0]

def PDE(self,x):
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

        loss_BC_1 = self.loss_function(f_BC_1(self,x_BC),torch.Tensor([x_0]).to(device))
        loss_BC_2 = self.loss_function(f_BC_2(self,x_BC),torch.Tensor([v_0]).to(device))

        return loss_BC_1 + loss_BC_2
    
    def lossPDE(self,x_PDE):

        g = x_PDE.clone()
        g.requires_grad = True
        f = self.forward(g)

        f_x = autograd.grad(f,g,torch.ones_like(f).to(device),retain_graph=True, create_graph=True)[0]
        f_xx = autograd.grad(f_x,g,torch.ones_like(f_x).to(device),retain_graph=True, create_graph=True)[0]

        index_1 = 2*bata
        index_2 = omega**2

        f_PDE = f_xx + index_1*f_x + index_2*f
        f_PDE = f_PDE[:,0]

        loss_PDE=self.loss_function(f_PDE.view(-1,1),PDE(self,x_PDE))

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

def accuracyRate(x,y):
    x = x.cpu().numpy()
    y = y.detach().cpu().numpy()
    yy = np.exp(-bata*x)*np.cos((omega**2-bata**2)**0.5*x)
    return np.sum(y-yy)
    

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
acclsit = []
acclistTemp = []

x_ACC = torch.tensor([0]).to(device)

for i in trange(steps):
    yh = model(x_BC)
    loss = model.loss(x_BC,x_PDE)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    x_ACC[0] = 0
    for j in range(10):
        acclistTemp.append(accuracyRate(x_ACC,model(x_ACC)))
        x_ACC[0] = x_ACC[0] + 1

    acclsit.append(np.sum(acclistTemp))
    losslist.append(loss.item())

    acclistTemp = []


xh = torch.linspace(0,10,1000).view(-1,1)
yh = model(xh.to(device))

bata_Real = bata
omega_Real = omega 

def Data(x):
    return 1 * np.exp(-bata_Real*x)*np.cos((omega_Real**2-bata_Real**2)**0.5*x)

x_Real = np.linspace(0,10,1000)

g = xh.to(device)
g = g.clone()
g.requires_grad = True
f = model(g)

f_x = autograd.grad(f,g,torch.ones_like(f))[0]

f_x_plot=f_x.detach().cpu().numpy()
yh_plot=yh.detach().cpu().numpy()

ax1 = plt.subplot(4,1,1)
ax2 = plt.subplot(4,1,2)
ax3 = plt.subplot(4,1,3)
ax4 = plt.subplot(4,1,4)

ax1.scatter(x,y,color='green',label='Data')
ax1.plot(xh,yh_plot[:,0],color='red',label='PINNs')
ax1.plot(x_Real,Data(x_Real),color='blue',label='Real')
ax1.set_xlabel('x',color='black')
ax1.set_ylabel('f(x)',color='black')
ax1.legend(loc = 'upper left')

ax2.plot(xh,f_x_plot,color='red',label='v')
ax2.set_xlabel('x',color='black')
ax2.set_ylabel('v(x)',color='black')

ax3.plot(np.linspace(0,len(losslist),len(losslist)),losslist)
ax3.set_xlabel("Iterate")
ax3.set_ylabel("Loss")
#ax3.set_ylim(0,1)

ax4.plot(np.linspace(0,len(acclsit),len(acclsit)),acclsit)
ax4.set_xlabel("Iterate")
ax4.set_ylabel("Accuracy")
#ax4.set_ylim(3,-3)

plt.show()