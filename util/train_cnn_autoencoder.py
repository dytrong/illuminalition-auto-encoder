import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from change_image_to_tensor import loader_dataset
from tqdm import tqdm

###reproducible
torch.manual_seed(1)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ######第一层卷积层
            nn.Conv2d(3, 64, 3, stride = 2, padding = 0), ### (N-k+2p)/s+1,向下取整 // 3*64*64 -> 64*31*31
            nn.BatchNorm2d(64),
            nn.PReLU(),
            ######第二层卷积层
            nn.Conv2d(64, 128, 3, stride = 2, padding = 0), ### 64*31*31 -> 128*15*15
            nn.BatchNorm2d(128),
            nn.PReLU(),
            ######第三层卷积层
            nn.Conv2d(128, 256, 3, stride = 2, padding = 0), ### 128*15*15 -> 256*7*7
            nn.BatchNorm2d(256),
            nn.PReLU(),
            ######第四层池化层
            nn.Conv2d(256, 512, 3, stride = 2, padding = 0), ### 256*7*7 -> 512*3*3
            nn.BatchNorm2d(512),
            nn.PReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride = 2, padding = 0), ## s*(N-1)+k-2p //4*4 -> 7*7
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 0), ## 7*7 -> 15*15
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 0), ## 15*15 -> 31*31
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.ConvTranspose2d(64, 3, 4, stride = 2, padding =0), ## 31*31 -> 64*64
            nn.BatchNorm2d(3),
            nn.PReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded.view(x.size(0), -1), decoded

#####加载模型
def autoencoder(model_parameters_path, pretrained=False, **kwargs):
    model = AutoEncoder(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_parameters_path))
    return model

def train(model, optimizer, loss_func, model_parameters_path, EPOCH, scheduler=None):
    #####加载数据
    train_loader, label_loader = loader_dataset()
    #####训练
    for epoch in tqdm(range(EPOCH)):
        #####训练
        model.train()
        #####学习率衰减
        if scheduler is not None:
            scheduler.step()
        #####初始化loss的值
        total_loss = 0
        train_step = 0
        #####一个for循环遍历两个列表
        for mini_train_data, mini_label_data in zip(train_loader, label_loader):
            optimizer.zero_grad()
            #########forward#########
            encoded,decoded = model(mini_train_data.cuda())
            loss = loss_func(decoded,mini_label_data.cuda())
            #########backward########
            loss.backward()
            total_loss = total_loss+loss.item()
            optimizer.step()
            train_step += 1
        if epoch % 5 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1,EPOCH,total_loss/train_step))
    ######保存模型参数
    torch.save(model.state_dict(), model_parameters_path)

if __name__ == "__main__":
    #####接收参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--EPOCH", type=int, required=True)
    parser.add_argument("--optimizer_method", type=str, choices=['Adam','SGD'], required=True)
    parser.add_argument("--model_parameters_path", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    #####加载模型
    model = AutoEncoder().cuda()
    #####选择优化方法
    if args.optimizer_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #####计算损失函数
    loss_func = nn.MSELoss()
    #####计算学习率衰减函数,每隔step_size个epoch就将学习率降为原来的gamma倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    #####训练
    train(model = model, 
          optimizer = optimizer, 
          loss_func = loss_func,
          model_parameters_path = args.model_parameters_path,
          EPOCH = args.EPOCH,
          scheduler = None)
    print("训练autoencoder模型参数共耗时:"+str(time.time()-start))
