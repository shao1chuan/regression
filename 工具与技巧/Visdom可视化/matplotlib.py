from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom()
import matplotlib.pyplot as plt
plt.plot([1, 23, 2, 4])
plt.ylabel('some numbers')
viz.matplot(plt)
