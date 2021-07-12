import torch
import torch.nn.functional as F
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
import matplotlib.pyplot

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.cov_unit = nn.Sequential(
            # x: [b,3,32,32]  => [b,6 随机梯度下降,28,28]
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            # out=(in - kernel_size +2 * padding) / stride + 1
        # [b, 6 随机梯度下降, 28, 28]  => [b,6 随机梯度下降,14,14]
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),  # 窗口2*2
            # [b, 6 随机梯度下降, 14, 14]  => [b,16,10,10]
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            # [b, 6 随机梯度下降, 10, 10]  => [b,16,5,5]
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # [b, 16, 5, 5]
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        # tmp = torch.randn(2,3,32,32)
        # out = self.cov_unit(tmp)
        # print("conv out",out.shape)
        # self.criteon = nn.CrossEntropyLoss()


    def forward(self,x):
        """

        :param x:[b,3,32,32]
        :return:
        """
        batchsize = x.size(0)
        # [b,3,32,32] => [b, 16, 5, 5]
        x = self.cov_unit(x)
        # [b,16,5,5] => [b, 16*5*5]
        x = x.reshape(batchsize,-1)
        # [b, 16*5*5] => [b,10]
        logits = self.fc_unit(x)
        # [b,10] 第二维 所以 dim=1
        # pred = F.softmax(logits,dim = 1)
        # loss = self.criteon(logits,y)
        return logits
def test():
    tmp = torch.randn(2,3,32,32)
    net = Lenet5()
    out = net(tmp)
    print("conv out",out.shape)
def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('../../../data/cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('../../../data/cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    model = Lenet5().to(device)
    print(f"神经网络结构 {model}")
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)

    epochs = 100
    losses = []
    for epoch in range(epochs):
        model.train()
        losssum = 0
        for bidx,(x,label) in enumerate(cifar_train):
            x,label = x.to(device),label.to(device)
            # logits [b,10]  label [b] loss tensor 标量
            logits = model(x)
            loss = criteon(logits,label)
            losssum += loss.item()
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(f"epoch is {epoch} loss is {losssum}")
        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for bidx,(x,label) in enumerate(cifar_test):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                # logits [b,10]
                pred = logits.argmax(dim = 1)
                # pred [b]
                total_correct += torch.eq(pred,label).float().sum().item()
                # torch.eq(pred, label)   [b] vs [b]
                total_num += x.size(0)
            acc = total_correct/total_num
            if epoch % 1 == 0:
                print(f"epoch is {epoch} acc is {acc}")
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(range(0, len(losses)), losses)
    matplotlib.pyplot.show()







if __name__ == '__main__':
    main()

# cd C:\Program Files\NVIDIA Corporation\NVSMI
# nvidia - smi
