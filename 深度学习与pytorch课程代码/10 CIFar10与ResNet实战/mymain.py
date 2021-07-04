import torch
from mylenet5 import Lenet5
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim

def loaddata(batchsz):

    cifar_train = datasets.CIFAR10('../../../use/data/cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('../../../use/data/cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)
    return cifar_train,cifar_test

def main():
    epochs = 100
    batchsz = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
    print("device is :" ,device)


    cifar_train,cifar_test = loaddata(batchsz)
    model = Lenet5().to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criteon = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        model.train()
        for batchszid ,(x,lable) in enumerate(cifar_train):
            # print(x.shape,lable.shape)
            x,lable = x.to(device),lable.to(device)
            logits = model(x)
            loss =criteon(logits,lable)
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"{epoch} loss :{loss} ")

        model.eval()
        with torch.no_grad():

            total_correct = 0
            total_num = 0
            for batchszid, (x, lable) in enumerate(cifar_test):
                # print(x.shape,lable.shape)
                x, lable = x.to(device), lable.to(device) #[b]
                logits = model(x)  #[b,10]
                pred = logits.argmax(dim = 1)#[b]
                correct = torch.eq(pred,lable).float().sum().item()
                total_correct +=correct
                total_num+=x.size(0)
            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)



if __name__ == '__main__':
    main()
