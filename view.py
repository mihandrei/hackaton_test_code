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

def plot_timeseries():
    data = read_out_array(sys.argv[1])

    for i in range(0,64,10):
        plt.plot(data[:, i , 0, 0])
        # plt.plot(data[:, i , 0, 2])
        # plt.plot(data[:, i , 0, 0] - data[:, i , 0, 3])


def plot_par_variance(node):
    data = read_out_array(sys.argv[1])
    plt.imshow(data[:, node, :], interpolation='nearest', extent=[0,6,0,6])

plot_par_variance(0)
# plot_timeseries()
plt.show()
