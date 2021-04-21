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

targetRisk = "E:/LifelongTeacherStudent/results/TeacherStudent_Bound_Target.txt"
trueRisk = "E:/LifelongTeacherStudent/results/TeacherStudent_TrueResults.txt"

f1 = open(targetRisk)
task1Data = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	task1Data.append(float(cNames[i]))
f1.close()
targetRisk = np.array(task1Data)

f1 = open(trueRisk)
task2Data = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	task2Data.append(float(cNames[i]))
f1.close()
trueRisk = np.array(task2Data)

import matplotlib.pyplot as plt

x1=range(1,21)
x2=range(11,21)

x11 = range(0,10)

x1 = np.zeros(40)
for i in range(40):
	x1[i] = i

x2 = np.zeros(40)
for i in range(40):
	x2[i] = i

for i in range(40):
    if i < 20:
        trueRisk[i] = 1 - trueRisk[i]
    else:
        trueRisk[i] = 2 - trueRisk[i]

plt.plot(x1,targetRisk,linewidth=3,color='b',marker='o',
         markerfacecolor='blue',markersize=1,label='L-TS-Lifelong')

plt.plot(x2,trueRisk,label='L-TS-JDT',linewidth=3,color='r',marker='*',
         markerfacecolor='red',markersize=3)

#plt.fill_between(x11,y1[0:10],y1[0:10],facecolor='green')
plt.grid(True)

plt.xlabel('Epochs')
plt.ylabel('Average risk')
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
