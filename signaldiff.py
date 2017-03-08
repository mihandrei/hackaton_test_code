import sys
import numpy as np
from matplotlib import pylab as plt

def read_arr(pth):
    data = [0] * 1024
    with open(pth) as f:
        for d in f.read().split():
            data.append(float(d))
    return np.array(data)

reference = read_arr('gccbuild/out.array')
reference =  np.nan_to_num(reference)
#print np.min(reference), np.max(reference), np.var(reference)


gpu_data = read_arr('pgibuild/out.array')
gpu_data =  np.nan_to_num(gpu_data)
#print np.min(gpu_data), np.max(gpu_data), np.var(gpu_data)


difference = np.max(np.abs(reference - gpu_data))

if difference == 0:
    print 'perfect match'

if difference < 1e-100:
    print 'all ok'
else:
    print 'abs difference is %f' % difference

