import numpy as np
import glob

from math import sqrt

def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab

def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den

def GetImagesByAttribute(attributeIndex=16):
    f = open("C:\CommonData\list_attr_celeba.txt")

    glass = []
    noGlass = []
    lines = f.readlines()
    del lines[0]
    del lines[0]

    for line in lines:
        array = line.split()

        if (array[attributeIndex] == "-1"):
            new_context = array[0]
            str1 = "C:\CommonData\img_celeba2\\"+new_context
            glass.append(str1)

        else:
            new_context = array[0]
            str1 = "C:\CommonData\img_celeba2\\"+new_context
            noGlass.append(str1)
    return glass,noGlass

def GetImagesByAttribute_Array(attributeArray,count):
    totalArr = []
    for i in range(len(attributeArray)):
        index = attributeArray[i]
        _,attributes = GetImagesByAttribute(index)
        attributes = attributes[0:count]
        totalArr.append(attributes)
    return totalArr

def MeanVector(zglass_p,znoglass_p,count):
    dim_z = 256
    tb1 = np.zeros(dim_z)
    tb2 = np.zeros(dim_z)
    batch_size = 64
    count2 = int(count / batch_size) * batch_size
    for i in range(dim_z):
        for j in range(count2):
            tb1[i] = tb1[i] + zglass_p[j, i]
            tb2[i] = tb2[i] + znoglass_p[j, i]

    # mean vector for glass to no glass
    tb1 = tb1 / count
    tb2 = tb2 / count
    return tb1,tb2

def MeanVector_One(zglass_p,count):
    dim_z = 256
    tb1 = np.zeros(dim_z)
    tb2 = np.zeros(dim_z)
    batch_size = 64
    count2 = int(count / batch_size) * batch_size
    for i in range(dim_z):
        for j in range(count2):
            tb1[i] = tb1[i] + zglass_p[j, i]

    # mean vector for glass to no glass
    tb1 = tb1 / count
    return tb1

def GetCeleBa_WithAttribute():

    # load dataset
    img_path = glob.glob('C:/commonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    n_examples = 202599

    f = open("C:\CommonData\list_attr_celeba.txt")

    glass = []
    noGlass = []
    lines = f.readlines()
    del lines[0]
    del lines[0]

    attribute = np.zeros((n_examples,40)).astype(int)
    index = 0
    for line in lines:
        array = line.split()
        del array[0]
        array = np.array(array).astype(int)
        attribute[index,:] = array
        index = index + 1

    return data_files,attribute

GetCeleBa_WithAttribute()