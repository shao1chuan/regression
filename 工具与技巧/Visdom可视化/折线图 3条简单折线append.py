import visdom
import numpy as np
import math
viz = visdom.Visdom()

viz.line([[0.,0.,0.]],[0.],win='折线图3',opts = dict(legend=['Y1', 'Y2','function'],title= '折线图3'))
for i in range(20):

    Y1 = 2*i
    Y2 = i**2
    Y3 = i
    viz.line([[Y1,Y2,0]],[i],win='折线图3',update='append')



