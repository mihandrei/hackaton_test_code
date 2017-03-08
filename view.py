from matplotlib import pylab as plt
import numpy as np
import sys

def read_out_array(pth):
    data = []

    with open(pth) as f:
        data_shape = tuple(int(d) for d in next(f).split())
        for line in f:
            for d in line.split():
                data.append(float(d))

    data = np.array(data)
    data = np.reshape(data, data_shape)

    print 'read  data of shape %s' % str(data_shape)
    return data

data = read_out_array(sys.argv[1])

for i in range(4):
    # plt.plot(data[:, i , 0, 1])
    plt.plot(data[:, i , 0, 2])
    plt.plot(data[:, i , 0, 0] - data[:, i , 0, 3])

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
