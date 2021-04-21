from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
import  numpy as np
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def Assign1(x_train,y_train,n_labeled):
    eachCount = int(n_labeled/10)

    index = 0
    count = [0,0,0,0,0,0,0,0,0,0,]
    train1 = []
    train2 = []
    train3 = []
    train4 = []
    train5 = []
    train6 = []
    train7 = []
    train8 = []
    train9 = []
    train10 = []

    real_y = []
    while True:
        index = random.randint(0, 50000)
        class1 = y_train[index]
        if count[class1] < eachCount:
            train = []
            if class1 == 0:
                train1.append(x_train[index])
            elif class1 == 1:
                train2.append(x_train[index])
            elif class1 == 2:
                train3.append(x_train[index])
            elif class1 == 3:
                train4.append(x_train[index])
            elif class1 == 4:
                train5.append(x_train[index])
            elif class1 == 5:
                train6.append(x_train[index])
            elif class1 == 6:
                train7.append(x_train[index])
            elif class1 == 7:
                train8.append(x_train[index])
            elif class1 == 8:
                train9.append(x_train[index])
            elif class1 == 9:
                train10.append(x_train[index])
            count[class1] = count[class1] + 1
            real_y.append(y_train[index])

        myCount = 0
        for a in range(10):
            if count[a] >= eachCount:
                myCount = myCount + 1
        if myCount >= 10:
            break
    real_y = []
    for p1 in range(10):
        for p2 in range(int(n_labeled/10)):
            real_y.append(p1)

    train1 = np.array(train1).astype(np.float32)
    train2 = np.array(train2).astype(np.float32)
    train3 = np.array(train3).astype(np.float32)
    train4 = np.array(train4).astype(np.float32)
    train5 = np.array(train5).astype(np.float32)
    train6 = np.array(train6).astype(np.float32)
    train7 = np.array(train7).astype(np.float32)
    train8 = np.array(train8).astype(np.float32)
    train9 = np.array(train9).astype(np.float32)
    train10 = np.array(train10).astype(np.float32)

    totalTrain = []
    totalTrain.append(train1)
    totalTrain.append(train2)
    totalTrain.append(train3)
    totalTrain.append(train4)
    totalTrain.append(train5)
    totalTrain.append(train6)
    totalTrain.append(train7)
    totalTrain.append(train8)
    totalTrain.append(train9)
    totalTrain.append(train10)

    return totalTrain,real_y
import math

def Create_encodered_dataset(encoder,x_train):
    n_number = np.shape(x_train)[0]
    codes,code2,code3 = encoder.predict(x_train)

    count1 = np.shape(codes)[0]
    a1 = np.random.randn(count1,np.shape(codes)[1])
    code2[:,:] = code2[:,:]*0.5
    code2 = np.exp(code2)

    code2[:,:] = code2[:,:]*a1[:,:]
    codes = codes + code2
    #codes = np.hstack((codes,code2))
    return codes


def Create_dataset_ALL(vae,encoder,train,totalNumber,each_class):
    count_class = int(totalNumber / each_class)
    train_total = []
    train_y = []
    real = []

    ass = train[0]
    total_count = int(count_class/each_class)

    for p3 in range(total_count):
        for p1 in range(10):
            a1 = train[p1]
            a1 = np.array(a1)
            a1 = vae.predict(a1)
            for p2 in range(np.shape(a1)[0]):
                real.append(a1[p2])
                train_y.append(p1)

    return real,train_y



def Create_dataset_ALL2(vae,encoder,decoder,train,totalNumber,each_class):
    count_class = int(totalNumber / each_class)
    train_total = []
    train_y = []
    real = []

    ass = train[0]
    total_count = int(count_class/each_class)

    for p3 in range(total_count):
        if p3 == 0:
            for p1 in range(10):
                a1 = train[p1]
                a1 = np.array(a1)
                a1 = vae.predict(a1)
                for p2 in range(np.shape(a1)[0]):
                    real.append(a1[p2])
                    train_y.append(p1)
        else:
            for p1 in range(10):
                a1 = train[p1]
                a1 = np.array(a1)
                a1 = vae.predict(a1)
                size = np.shape(a1)[0]

                pp = np.random.normal(0, 1, np.shape(a1)[0]*np.shape(a1)[1]*np.shape(a1)[2])
                pp = pp / 0.000001
                
                a1 = a1 + pp

                for p2 in range(np.shape(a1)[0]):
                    real.append(a1[p2])
                    train_y.append(p1)

    return real,train_y



def Create_dataset(encoder,train,totalNumber,each_class):
    count_class = int(totalNumber / each_class)
    train_total = []
    train_y = []
    real = []
    for class1 in range(count_class):
        a = random.randint(0, 9)
        train1 = train[a]
        z_mean, z_log_var, z_code = encoder.predict(train1)

        count1 = np.shape(z_mean)[0]
        a1 = np.random.randn(count1, np.shape(z_mean)[1])
        z_log_var[:, :] = z_log_var[:, :] * 0.5
        z_log_var = np.exp(z_log_var)

        z_log_var[:, :] = z_log_var[:, :] * a1[:, :]
        z_code = z_mean + z_log_var

        for t1 in range(np.shape(z_code)[0]):
            train_total.append(z_code[t1])
            real.append(train1[t1])
            train_y.append(a)

    return train_total,train_y,real

