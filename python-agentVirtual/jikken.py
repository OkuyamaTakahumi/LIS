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


q = np.array([0,0,0])
max_q_abs = max(abs(q))
if max_q_abs > 0:
    q = q / float(max_q_abs)


RegularQ = q / float(max_q_abs)
RegularQ


max_q_abs
q

'''
import numpy as np
import matplotlib.pyplot as plt



def pause_Q_plot(q):


    actions = [0, 1, 2]

    plt.cla()

    plt.xticks([0, 1, 2])
    plt.xlabel("Action") # x軸のラベル
    plt.ylabel("Q_Value") # y軸のラベル
    plt.ylim(0, 1.5)  # yを0-5000の範囲に限定
    plt.xlim(-0.5, 2.5) # xを2-15の範囲に限定


    plt.bar(actions,q,align="center")

    # - plt.show() ブロッキングされてリアルタイムに描写できない
    # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
    # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
    plt.pause(1.0 / 10**30)
    #plt.show()


if __name__ == "__main__":

    #fig, ax = plt.subplots(1, 1)

    #ax.bar(actions,np.random.rand(3),align="center")
    #q_now = np.load('./q_now.npy')
    #print q_now.ravel()

    for i in range(50):
        print i
        q_now = np.random.rand(3)
        pause_Q_plot(q_now.ravel())
