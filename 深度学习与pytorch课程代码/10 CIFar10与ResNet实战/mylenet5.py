import torch
from torch import nn
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,stride=1,padding=0,kernel_size=5),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, stride=1, padding=0, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            nn.Linear(in_features=16*5*5,out_features=120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
        )
        # testdata = torch.randn(2,3,32,32)
        # # torch.Size([2, 16, 5, 5])
        # print(self.net(testdata).shape)

    def forward(self,x):
        return self.net(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Lenet5()
    model.to(device)
    inputs = torch.randn(2, 3, 32, 32)
    inputs = inputs.to(device)
    print(model(inputs).shape)
    print(model)

if __name__ == '__main__':
    main()