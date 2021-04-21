from keras.datasets import mnist
import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
import keras as keras
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def ReadFiles(file):
	f1 = open(file)
	myData = []
	cNames = f1.readlines()
	for i in range(0, len(cNames)):
		myData.append(float(cNames[i]))
	f1.close()
	myData = np.array(myData)
	return myData

mnist = "E:/LifelongTeacherStudent/results/LongTask_ELBOs.txt"

f1 = open(mnist)
task1Data = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	task1Data.append(float(cNames[i]))
f1.close()
mnistData = np.array(task1Data)

x1 = np.zeros(120)
for i in range(120):
	x1[i] = i

x1 = np.array(x1)

import matplotlib.pyplot as plt

plt.plot(x1,mnistData,linewidth=3,color='b',marker='o',
         markerfacecolor='blue',markersize=1,label='L-TS-MNIST')

#plt.fill_between(x11,y1[0:10],y1[0:10],facecolor='green')
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.title('Catastrophic forgetting during the lifelong learning')
plt.legend()
'''
plt.text(7.5,
	0.75,
	"MNIST(Old)",
	fontsize=15,
	verticalalignment="top",
	horizontalalignment="right"
)

plt.text(20,
	1.75,
	"MNIST-Fashion(New)",
	fontsize=15,
	verticalalignment="top",
	horizontalalignment="right"
)
'''
plt.show()
