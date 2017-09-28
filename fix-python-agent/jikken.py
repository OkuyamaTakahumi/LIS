# coding:utf-8
'''
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

'''
class Num(object):
    def __init__(self,n):
        self.num = n
number = Num(5)
print number
print Num(4)

print number.num

import numpy as np
ilsvrc = np.load('/Users/okuyamatakashi/lis/fix-python-agent/ilsvrc_2012_mean.npy')

ilsvrc.shape
print "bbb" ,
print "aaa"

def hoge():
    return 1,2

a, = hoge()
print a

a = 3.123456678910
3 * (a < 6)
3 * (a > 6)
print "%.3f"%(a)

np.mod(9393929394939912,10)
