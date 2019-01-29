# test resample
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
a = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
x = np.linspace(0,1,5)
y = np.linspace(0,1,5)
# assume pixel become smaller
dx1 = 0.5
dy1 = 0.5
