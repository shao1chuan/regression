import  torch
from    torch import optim, nn
import  visdom
import  torchvision
from    torch.utils.data import DataLoader

from    lemon import Lemon
from    resnet import ResNet18



batchsz = 32
lr = 1e-3
epochs = 10

device = torch.device('cpu')
torch.manual_seed(1234)
root = "data/lemon/"

train_db = Lemon(root, 224, mode='train')
val_db = Lemon(root, 224, mode='val')
test_db = Lemon(root, 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)



def evalute(model, loader):
    model.eval()
    
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def main():

    model = ResNet18(4).to(device)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)

if __name__ == '__main__':
    main()
