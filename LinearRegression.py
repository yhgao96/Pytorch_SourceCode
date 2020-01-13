#pytorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


#fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)  #将1维数据转化为2维
y=x.pow(2)+0.2*torch.rand(x.size())

# plt.scatter(x,y)
# plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=nn.Linear(n_feature,n_hidden)   #自己搭建的层作为这个类的一个属性(input,output)
        self.predict=nn.Linear(n_hidden,n_output)   #另外一层（input,output）

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

plt.ion()
plt.show()

net=Net(1,10,1)   #n_feature,n_hidden,n_output
print(net)        #打印出网络结构

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)     #优化器(传入神经网络中的参数  学习率lr)
loss_func=nn.MSELoss()                                 #损失函数（均方误差）

for t in range(500):
    prediction=net(x)

    loss=loss_func(prediction,y)
    optimizer.zero_grad()   #把神经网络中的梯度设为0
    loss.backward()
    optimizer.step()

    if t%5==0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(t + 1, 500, loss.item()))
        plt.cla()
        plt.scatter(x.numpy(),y.numpy())
        plt.plot(x.numpy(),prediction.detach().numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.item(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
print('Done!')

'''
# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

plt.ion()
plt.show()

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        # Plot the graph
        plt.cla()
        predicted = model(torch.from_numpy(x_train)).detach().numpy()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, predicted, label='Fitted line')
        plt.text(8,2, 'Loss=%.4f' % loss.item(), fontdict={'size': 15, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
'''