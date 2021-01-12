# y = (x-3)**2+5

rate = 0.01
x = 10

def step(x,rate):
    return x - 3*(x-3)**2*rate
for i in range(0,100):
    x= step(x,rate)
    print(x)

y = (x-3)**2+5
print(y)