##  zq


import numpy as np
Lr=0.1
Batch_size =2

Node_num = np.array([3, 10, 10, 1])  #各层神经元数，输入层 ，隐藏层，输出层
#输入层 + 隐藏层 + 输出层数
Layer_num = Node_num.size

#########################################################################################
# w0:l0->l1层的权重， w1：l1->l2层的权重
W = []
for i in range(Layer_num-1):
    W.append(np.random.random([ Node_num[i], Node_num[i+1] ]))
#print(W)

B = [] #偏移量，B[0]是用不上的
for i in range( Layer_num ):
    B.append(np.random.random(Node_num[i]))
#print("B:",B)

#输入层 + 隐含层 + 输出层 节点
Node = []
for i in range( Layer_num):
    Node.append(np.zeros([Batch_size, Node_num[i]]))
print("Node:",Node)

#########################################################################################
#记录loss函数对各个w的偏导数
W_grad = []
for i in range(Layer_num-1):
    W_grad.append(np.zeros([ Node_num[i], Node_num[i+1] ]))
#print(W_grad)

B_grad = []
for i in range( Layer_num ):
    B_grad.append(np.zeros(Node_num[i]))
#print("B_grad:", B_grad)

# 记录loss函数对各个神经元节点值得偏导数
Node_grad = []
for i in range(Layer_num):
    Node_grad.append(np.zeros([Batch_size, Node_num[i]]))
print("Node_grad",Node_grad)

########################################################################################
def forward(input):
    Node[0]=input
    for i in range( Layer_num-1):
        Node[i+1] = Node[i]@W[i] + B[i+1]  #更新各层
        Node[i+1] = 1 / ( 1+np.exp(-Node[i+1]) )   #都采用sigmoid激活函数
    #Node[Layer_num-1] = Node[Layer_num-2]@W[Layer_num-2] + B[Layer_num-1]
    return Node[Layer_num-1] #将输出层输出

def lossFunction(output, y):
    return np.sum( (output - y)**2 )  #函数形式可以改变

def backward(y):
    #输出层梯度Node_grad[Layer_num-1]单独处理
    delta = 0.0001 #用数值方法计算输出层节点偏导数
    loss = lossFunction(Node[Layer_num-1], y)
    #print("loss:",loss)
    for i in range(Node_num[Layer_num - 1]): #输出层节点数
        B_grad[Layer_num - 1][i] = 0.0
        for b in range(Batch_size):
            output = Node[Layer_num-1]
            output[b,i]+=delta #给输出层第 i 个节点一个小增量，计算loss的变化，由此得到loss对输出层第 i 节点偏导数，更新Node_grad[Layer_num-1][i]
            #print("output:",output)
            #print("hahaha:",Node_grad[Layer_num-1][0,0])
            Node_grad[Layer_num-1][b,i] = (lossFunction(output, y) - loss )/delta #更新Node_grad的最后一层
            B_grad[Layer_num-1][i] += Node[Layer_num-1][b,i]*(1-Node[Layer_num-1][b,i]) * Node_grad[Layer_num-1][b,i] #B_grad每个元素跟Node, Node_grad 相应元素都有这样的关系

    #更新隐含层的 Node_grad，W_grad, B_grad
    for l in range( Layer_num-2, 0, -1 ): # l表示层，从后向前更新
        #更新W_grad[ Node_num[l], Node_num[l+1] ]
        for i in range( Node_num[l] ):
            for j in range (Node_num[l+1] ):
                W_grad[l][i,j] = 0.0
                for b in range(Batch_size):
                    W_grad[l][i,j] += Node[l][b,i] * Node[l+1][b,j]*(1-Node[l+1][b,j]) * Node_grad[l+1][b,j]
        #更新Node_grad
        for i in range( Node_num[l] ):
            B_grad[l][i] = 0.0
            for b in range( Batch_size ):
                Node_grad[l][b,i] = 0.0
                for j in range(Node_num[l+1]):
                    Node_grad[l][b,i] += W[l][i,j] * Node[l+1][b,j]*(1-Node[l+1][b,j]) * Node_grad[l+1][b,j]
                    B_grad[l][i] += Node[l][b,i]*(1-Node[l][b,i]) * Node_grad[l][b,i]

    #更新W_grad[0]
    for i in range( Node_num[0] ):
        for j in range (Node_num[1] ):
            W_grad[0][i,j] = 0.0
            for b in range(Batch_size):
                W_grad[0][i,j] += Node[0][b,i] * Node[1][b,j]*(1-Node[1][b,j]) * Node_grad[1][b,j]

def step():
    for l in range(Layer_num-1):
        for i in range( Node_num[l] ):
            for j in range( Node_num[l+1] ):
                W[l][i,j] -= Lr * W_grad[l][i,j]
    for l in range(1, Layer_num):
        for i in range( Node_num[l]):
            B[l][i] -= Lr*B_grad[l][i]
#test
#测试数据
test_x = np.array([[1,2,3],[4,5,6]])
test_y = np.array([[0.4],[0.7]])
#print("Node:",Node)
for i in range(100000):
    output = forward(test_x)
    loss=lossFunction(output, test_y)
    print("output:",output,"loss:",loss)
    backward(test_y)
    step()

