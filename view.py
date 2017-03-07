from matplotlib import pylab as plt
import matplotlib.cm as cm
import numpy as np
import sys

data = []

with open(sys.argv[1]) as f:
    for d in f.read().split():
        data.append(float(d))

data = np.array(data)

print len(data)
data = np.reshape(data, (300, 256, 1, 6))


plt.plot(data[:, 150 , 0, :])

# data = data[: , : , 0, 0]
##-----
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ims = ax.imshow(data, cmap=cm.jet, interpolation='nearest')
# fig.colorbar(ims)
#
# numrows, numcols = data.shape
#
# def format_coord(x, y):
#     col = int(x+0.5)
#     row = int(y+0.5)
#     if col>=0 and col<numcols and row>=0 and row<numrows:
#         z = data[row,col]
#         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
#     else:
#         return 'x=%1.4f, y=%1.4f'%(x, y)
#
# ax.format_coord = format_coord
plt.show()
