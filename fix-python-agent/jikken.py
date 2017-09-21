# coding:utf-8
a = 2
print 'Hello World'
import numpy as np
array =np.array([2,3])
print type(a)
print type(array)

image = np.load('./CameraImages/image.npy')
image.shape

import sys

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(image)
