import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from myresnet import Resnet18
from torch import nn,optim
import  visdom
def evalute(model,loader):
    model.eval()
    correct = 0
    total_num = len(loader.dataset)
    for idx, (x, label) in enumerate(loader):
        x, label = x.to(device), label.to(device)
        logits = model(x)
        # print(logits.shape,label.shape)
        # torch.Size([128, 10]) torch.Size([128])
        pred = logits.argmax(dim=1)
        # print(pred.shape)
        # torch.Size([128])
        correct += torch.eq(pred, label).sum().item()
    return correct / total_num


def loaddata():
    batchsz = 128

    cifar_train = datasets.CIFAR10(root, True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    cifar_test = datasets.CIFAR10(root, False, transform=transforms.Compose([
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

    model = model.to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    global_step = 0
    best_acc, best_epoch = 0, 0

    total_num = len(cifar_test.dataset)
    for epoch in range(epochs):

        for step,(x,label) in enumerate(cifar_train):
            x,label = x.to(device),label.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step+=1
        print(f"{epoch}, loss is {loss.item()}")

        val_acc = evalute(model,cifar_test)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc

            torch.save(model.state_dict(), 'best.mdl')
            # torch.save(root, 'best.mdl')

        viz.line([val_acc], [epoch], win='acc', update='append')
        print(f"{epoch} ,acc is {val_acc}")

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, cifar_test)
    print('test acc:', test_acc)








if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = "../../../../use/data/cifar"
    main()