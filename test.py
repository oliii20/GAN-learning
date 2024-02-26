import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

#准备数据
#对数据做归一化 （-1，1） 在GAN当中需要
transform = transforms.Compose([
    transforms.ToTensor(),          ##channel在前high和width在后,转换到01
    transforms.Normalize(0.5, 0.5)  #
])
train_ds = torchvision.datasets.MNIST('data', train=True, transform = transform, download = True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle= True)
#Minst里数据维度（1，28，28）
################定义生成器 用tanh激活 在（-1，1）############
#输入是长度为100的噪声（正态分布随机数）
#输出为（1，28，28）的图片
#linear 1 :  100---256
#linear 2 :  256---512
#linear 3 :  512---28*28
#reshape : 28*28 ---(1,28,28)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):         #x 表示长度为100的noise输入
        img = self.main(x)
        img = img.view(-1, 28, 28, 1) #形状为 (batch_size, 28, 28, 1)。  其中 -1 表示自动推断该维度的大小，以使得总元素个数不变。
        return  img
######################定义判别器
#输入为（1，28，28）的图片，输出为二分类的概率值，输出使用sigmoid激活 0-1
#BCEloss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(), #负值保留梯度
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.main(x)
        return x

###
#初始化模型、优化器及损失计算函数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)  ###模型初始化
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
loss_fn = torch.nn.BCELoss()  #二元交叉熵损失

#绘图函数
def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.ion()
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i] + 1)/2)
        plt.axis('off')
    plt.show()
    plt.ioff()
test_input  = torch.randn(16, 100, device = device) #16个长度为100

#GAN的训练
D_loss = []
G_loss = []
#训练循环
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) #返回批次数 len(dataset)返回样本数
    for step, (img,_) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()
        real_output = dis(img)   #判别器输入真实的图片， real_output是对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output))  #判别器在真实图像上的损失
        d_real_loss.backward()

        gen_img = gen(random_noise) \
        # 判别器输入生成的图片， fake_output对生成图片的预测
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output))  #判别器在生成图像上的损失
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)  #不做梯度的截断
        g_loss = loss_fn(fake_output,
                              torch.ones_like(fake_output))  #生成器的损失
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)
            print('Epoch:', epoch)
            gen_img_plot(gen, test_input)
