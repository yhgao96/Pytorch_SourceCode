import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)         #class x data (tensor),shape=(100,2)
y0 = torch.zeros(100)                 #class y data  (tensor),shape=(100,1)
x1 = torch.normal(-2*n_data,1)        #class x data (tensor),shape=(100,2)
y1 = torch.ones(100)                  #class y data (tensor),shape=(100,1)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0)
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(2,10,2)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()

plt.ion()

for i in range(100):
    out = net(x)

    loss=loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2 ==0:
        plt.cla()
        # 获取概率最大的类别的索引
        prediction = torch.max(F.softmax(out),1)[1]
        # 将输出结果变为一维
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1], c = pred_y, s=100, lw=0)
        # 计算准确率
        accuracy = sum(pred_y==target_y)/200
        plt.text(1.5,-4,'Accuracy=%.2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
print('Done')