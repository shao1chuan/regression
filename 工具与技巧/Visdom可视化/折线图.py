import visdom
import numpy as np
import math
viz = visdom.Visdom()

X1 = np.linspace(-10,10,20)
Y1 = X1**2-3
Y2 = 4*X1+4
Y = np.column_stack((Y1,Y2))
X = np.column_stack((X1,X1))
print(X)

viz.line(Y,X,win='折线图1' ,
         opts=dict(title = 'title',
        legend=['aaaaaa','bbbbbb'],
        xlabel='xx',
        ylalel='yy'))

X = [i for i in range(-10,10)]
Y = [[x*2,x*3] for x in X]
viz.line(Y,X,win='折线图2',opts = dict(title= '折线图2'))




