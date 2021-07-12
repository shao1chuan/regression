import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from myresnet import Resnet18
from torch import nn,optim
import  visdom

def loaddata():
    batchsz = 128

    cifar_train = datasets.CIFAR10('../../../../use/data/cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    cifar_test = datasets.CIFAR10('../../../../use/data/cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    return cifar_train,cifar_test
def main():
    cifar_train,cifar_test = loaddata()
    # x,label = iter(cifar_train).next()
    # print(x.shape,label.shape)
    # torch.Size([128, 3, 32, 32])
    # torch.Size([128, 3, 32, 32]) torch.Size([128])
    viz = visdom.Visdom()
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='acc', opts=dict(title='val_acc'))

    epochs = 5
    model = Resnet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for step,(x,label) in enumerate(cifar_train):
            x,label = x.to(device),label.to(device)
            logits = model(x)
            loss = criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step+=1
        print(f"{epoch}, loss is {loss.item()}")


        model.eval()
        total_correct = 0
        total_num = len(cifar_test.dataset)

        for idx,(x,label) in enumerate(cifar_test):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # print(logits.shape,label.shape)
            # torch.Size([128, 10]) torch.Size([128])
            pred = logits.argmax(dim=1)
            # print(pred.shape)
            # torch.Size([128])
            correct = torch.eq(pred,label).sum().item()
            total_correct+=correct
            acc = total_correct / total_num
            viz.line([acc], [epoch], win='acc', update='append')
        print(f"{epoch} ,acc is {acc}")








if __name__ == '__main__':
    main()