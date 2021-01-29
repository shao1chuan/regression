import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../../../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


torch.manual_seed(1)
w1, b1 = torch.randn(200, 784, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True),\
         torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


# def f1(x,w,b):
#     return x@w.t() + b
class F1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,w,b):
        # x:[210,784],  w:[200,784], b[200,1]
        ctx.save_for_backward(x,w,b)
        print(f"开始正向传播")
        X =  x@w.t() + b
        # X torch.Size([210, 200])
        return X,w,b

    @staticmethod
    def backward(ctx, grad_x,grad_w,grad_b):
        x,w,b = ctx.saved_tensors
        # error = np.dot(params['W3'].T, error) * self.sigmoid(params['X2'], derivative=True)
        # change_w['W2'] = np.outer(error, params['O1'])
        # grad_w,grad_x,grad_b = grad_output
        grad_x_new = grad_x * w.t()
        # [210,200]

        grad_w_new = torch.mm(grad_x.T * x)

        grad_b_new = grad_b

        print(f"开始反向传播 grad_x is {grad_x}")
        return grad_x_new,grad_w_new,grad_b_new

# class F1Layer(nn.Module):
#     def forward(self, x,w1,b1):
#         x,w1, b1 = F1.apply(O0, w1, b1)
#         return x

# def forward(x):
#     x = x@w1.t() + b1
#     x = F.relu(x)
#     x = x@w2.t() + b2
#     x = F.relu(x)
#     x = x@w3.t() + b3
#     x = F.relu(x)
#     return x



optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        O0 = data
        X1,w1,b1 = F1.apply(O0,w1,b1)
        O1 = F.relu(X1)
        X2,w2, b2 = F1.apply(O1, w2, b2)
        O2 = F.relu(X2)
        X3,w3, b3 = F1.apply(O2, w3, b3)
        O3 = F.relu(X3)
        logits = O3
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data = data.view(-1, 28 * 28)
        O0 = torch.tensor(data)
        X1 = F1.apply(O0, w1, b1)
        O1 = F.relu(X1)
        X2 = F1.apply(O1, w2, b2)
        O2 = F.relu(X2)
        X3 = F1.apply(O2, w3, b3)
        O3 = F.relu(X3)
        logits = O3
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
