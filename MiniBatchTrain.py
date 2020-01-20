'''Reference Morvan Python
'''
import torch
import torch.utils.data as Data

Batch_SIZE = 8
x = torch.linspace(1, 10, 10)       #this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       #this is y data (torch tensor)
print(x)
print(y)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=Batch_SIZE,
    shuffle=True                    #True打乱样本     False按照顺序不打乱样本
    #num_workers=2,
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        #training...
        print('Epoch: ',epoch,'| step: ',step,'| batch x: ',batch_x.numpy(),'| batch y:',batch_y.numpy())

        
