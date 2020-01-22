# library
# standard library
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./mnist',train=True,
                                       transform=transforms.ToTensor(),   #(0-1)    (0-255)
                                       download=DOWNLOAD_MNIST
                                       )

# #plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[1].numpy(),cmap='gray')
# plt.title('%i' % train_data.train_labels[1])
# plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST(root='./mnist/',train=False,transform=transforms.ToTensor())      #FALSE说明是testdata
test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.  #
test_y=test_data.test_labels[:2000]
# test_loader=orch.utils.data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(                    #(1,28,28)
                in_channels=1,            #inputchannel,黑白图片
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,        #if stride = 1 ,padding = (kernel_size-1)/2=(5-1)/2
            ),      #->(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #->(16,14,14)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),          #->(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),             #->(32,7,7)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)        #batch(32,7,7)
        x=x.reshape(x.size(0),-1)     #batch(32*7*7)
        output=self.out(x)
        return output

cnn=CNN()
print(cnn)    #net architecture

optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()   #optimize all cnn parameyers

#training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):

        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50==0:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.numpy()
            # accuracy=sum(pred_y==test_y)/test_y.size(0)
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

print('Done!')
#print 10 prediction from test data
test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number') 



