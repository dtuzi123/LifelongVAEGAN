from Basic_structure import *
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

os.environ['CUDA_VISIBLE_DEVICES']='6'

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)

        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def My_Encoder_mnist(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def My_Classifier_mnist(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def MINI_Classifier(s, scopename,reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1,y


# Create model of CNN with slim api
def Image_classifier(inputs,scopename, is_training=True,reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 28, 28, 1])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 32
        self.len_discrete_code = 4
        self.epoch = 10

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        data_X, data_y = load_mnist(mnistName)
        x_train = data_X[0:60000]
        x_test = data_X[60000:70000]
        y_train = data_y[0:60000]
        y_test = data_y[60000:70000]

        self.mnist_train_x = x_train
        self.mnist_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.mnist_train_y[:, 0] = 1
        self.mnist_label = y_train
        self.mnist_label_test = y_test

        self.mnist_test_x = x_test
        self.mnist_test_y = y_test

        data_X, data_y = load_mnist(fashionMnistName)

        x_train1 = data_X[0:60000]
        x_test1 = data_X[60000:70000]
        y_train1 = data_y[0:60000]
        y_test1 = data_y[60000:70000]

        self.mnistFashion_train_x = x_train1
        self.mnistFashion_train_y = np.zeros((np.shape(x_train1)[0], 4))
        self.mnistFashion_train_y[:, 1] = 1
        self.mnistFashion_label = y_train1

        self.mnistFashion_test_x = x_test1
        self.mnistFashion_test_labels = y_test1

        '''
        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = np.shape(data_files)[0]
        batch = [get_image(batch_file, 105, 105,
                           resize_height=28, resize_width=28,
                           crop=False, grayscale=True) \
                 for batch_file in data_files]

        thirdX = np.array(batch)

        for t1 in range(n_examples):
            a1 = thirdX[t1]
            for p1 in range(28):
                for p2 in range(28):
                    if thirdX[t1, p1, p2] == 1.0:
                        thirdX[t1, p1, p2] = 0
                    else:
                        thirdX[t1, p1, p2] = 1

        myTest = thirdX[0:self.batch_size]
        self.thirdX = np.reshape(thirdX, (-1, 28, 28, 1))
        self.thirdY = np.zeros((np.shape(self.thirdX)[0], 4))
        self.thirdY[:, 2] = 1

        # ims("results/" + "gggg" + str(0) + ".jpg", merge(myTest[:64], [8, 8]))
        '''
        cc1 = 0

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])

        # GAN networks
        gan_code = tf.concat((self.z, self.y), axis=1)
        G1 = Generator_mnist("GAN_generator", gan_code, reuse=False)

        ## 1. GAN Loss
        # output of D for real images
        D_real, D_real_logits, _ = Discriminator_Mnist(self.inputs, "discriminator", reuse=False)

        # output of D for fake images
        D_fake, D_fake_logits, input4classifier_fake = Discriminator_Mnist(G1, "discriminator", reuse=True)

        self.g_loss = tf.reduce_mean(D_fake_logits)
        self.d_loss = tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G1
        _, d_hat, _ = Discriminator_Mnist(x_hat, "discriminator", reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx

        # losses
        '''
        d_r_loss = tf.losses.mean_squared_error(tf.ones_like(D_real_logits), D_real_logits)
        d_f_loss = tf.losses.mean_squared_error(tf.zeros_like(D_fake_logits), D_fake_logits)
        self.d_loss = (d_r_loss + d_f_loss) / 2.0
        self.g_loss = tf.losses.mean_squared_error(tf.ones_like(D_fake_logits), D_fake_logits)
        '''
        """ Graph Input """
        # images

        self.isPhase = 0

        # domain 1
        z_mean, z_log_sigma_sq = My_Encoder_mnist(self.inputs, "encoder1", batch_size=64, reuse=False)
        out_logit, softmaxValue = My_Classifier_mnist(self.inputs, "classifier", batch_size=64, reuse=False)

        continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        
        #define the classifier
        label_logits = Image_classifier(self.inputs, "Mini_classifier", reuse=False)
        self.mini_classLoss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=label_logits, labels=self.labels))

        log_y = tf.log(softmaxValue + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        log_y2 = tf.log(tf.nn.softmax(label_logits) + 1e-10)
        discrete_real2 = my_gumbel_softmax_sample(log_y2, np.arange(10))

        y_labels = tf.argmax(softmaxValue, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        code1 = tf.concat((continous_variables, discrete_real, discrete_real2), axis=1)
        reco1 = Generator_mnist("generator1", code1, reuse=False)
        reco2 = reco1

        # VAE loss
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean - y_labels) + tf.square(z_log_sigma_sq) - tf.log(1e-8 + tf.square(z_log_sigma_sq)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        self.vae_loss1 = reconstruction_loss1 + KL_divergence1

        self.vaeLoss = self.vae_loss1

        # classification loss
        self.classifier_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_logit, labels=self.y))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder1')]
        encoderClassifier_vars1 = [var for var in T_vars if var.name.startswith('classifier')]
        generator1_vars = [var for var in T_vars if var.name.startswith('generator1')]
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars = [var for var in T_vars if var.name.startswith('GAN_generator')]
        MiniClassifier_vars = [var for var in T_vars if var.name.startswith('Mini_classifier')]

        self.output1 = reco1
        self.output2 = reco2
        self.GAN_output = G1

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.vae1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=encoder_vars1 + generator1_vars)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=discriminator_vars1)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=GAN_generator_vars)
            self.classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classifier_loss, var_list=encoderClassifier_vars1)
            self.mini_classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.mini_classLoss, var_list=MiniClassifier_vars)

        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "Mini_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def Give_predictedLabels(self,testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions)
        return totalPredictions

    def Calculate_accuracy(self,testX,testY):
        #testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def test(self):
        with tf.Session() as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion')

            myPredict = self.predict()

            myIndex = 2

            testX = np.concatenate((self.mnist_train_x,self.mnistFashion_train_x),axis=0)
            testY = np.concatenate((self.mnist_train_y,self.mnistFashion_train_y),axis=0)
            index = [i for i in range(np.shape(testX)[0])]
            random.shuffle(index)
            testX = testX[index]
            testY = testY[index]
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            g_outputs = self.sess.run(
                self.GAN_output,
                feed_dict={self.inputs: testX[0:self.batch_size], self.z: batch_z, self.y: testY[0:self.batch_size]})

            g_outputs = np.reshape(g_outputs,(-1,28,28,1))
            ims("results/" + "mnistToFashion_gan" + str(0) + ".png", merge(g_outputs[:36], [6, 6]))

            real1 = testX[0:self.batch_size]
            my11 = np.reshape(real1,(-1,28,28,1))
            ims("results/" + "mnistToFashion_real" + str(0) + ".png", merge(my11[:36], [6, 6]))

            my11 = self.sess.run(self.output1, feed_dict={self.inputs: testX[0:self.batch_size]})
            my11 = np.reshape(my11,(-1,28,28,1))
            ims("results/" + "mnistToFashion_reco" + str(0) + ".png", merge(my11[:36], [6, 6]))

            mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
            mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x, self.mnistFashion_test_labels)

            z_mean, z_log_sigma_sq = My_Encoder_mnist(self.inputs, "encoder1", batch_size=64, reuse=True)
            continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

            z_in = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])

            code1 = tf.concat((z_in, self.y, self.labels), axis=1)
            reco1 = Generator_mnist("generator1", code1, reuse=True)

            myTest_x = self.mnist_train_x[0:self.batch_size]
            myTest_y = self.mnist_train_y[0:self.batch_size]
            myTest_label = self.mnist_label[0:self.batch_size]

            myindex = 10
            myTest_x = self.mnistFashion_train_x[myindex*self.batch_size:(myindex+1)*self.batch_size]
            myTest_y = self.mnistFashion_train_y[myindex*self.batch_size:(myindex+1)*self.batch_size]
            myTest_label = self.mnistFashion_label[myindex*self.batch_size:(myindex+1)*self.batch_size]

            minx = -2.0
            minx = 0
            diff = 4.0 / 64.0
            for i in range(32):
                myNew = []
                myCodes = sess.run(continous_variables,feed_dict={self.inputs:myTest_x})
                for j in range(64):
                    myCodes[0,i] = minx + j * diff
                    reco = sess.run(reco1, feed_dict={z_in: myCodes,self.y:myTest_y,self.labels:myTest_label})
                    myNew.append(reco[0])

                myNew = np.array(myNew)
                myNew = np.reshape(myNew,(-1,28,28))
                ims("results/" + "PPP" + str(i) + ".png", merge(myNew[:64], [8, 8]))

            bc = 0

    def Generate_GAN_Samples(self, n_samples, classN):
        myArr = []
        for tt in range(classN):
            y1 = np.zeros((self.batch_size, 4))
            y1[:, 0] = 1
            num1 = int(n_samples / self.batch_size)
            for i in range(num1):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                g_outputs = self.sess.run(
                    self.GAN_output,
                    feed_dict={self.z: batch_z, self.y: y1})
                for t1 in range(self.batch_size):
                    myArr.append(g_outputs[t1])

        myArr = np.array(myArr)
        return myArr

    def train_classifier(self):
        isFirstStage = True
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion')

            # saver to save model
            self.saver = tf.train.Saver()

            old_Nsamples = 60000
            oldX = self.Generate_GAN_Samples(old_Nsamples, 1)
            oldY = np.zeros((np.shape(oldX)[0], 4))
            oldY[:, 0] = 1

            oldLabels = self.Give_predictedLabels(oldX)
            oldX = oldX[0:np.shape(oldLabels)[0]]
            oldY = oldY[0:np.shape(oldLabels)[0]]

            # define combination of old and new datasets
            # second state
            dataX = np.concatenate((self.mnistFashion_train_x, oldX), axis=0)
            dataY = np.concatenate((self.mnistFashion_train_y, oldY), axis=0)
            dataLabels = np.concatenate((self.mnistFashion_label, oldLabels),axis=0)
            labelsY = dataLabels

            # third stage
            # dataX = np.concatenate((self.thirdX, oldX), axis=0)
            # dataY = np.concatenate((self.thirdY, oldY), axis=0)

            # First stage
            if isFirstStage:
                dataX = self.mnist_train_x
                dataY = self.mnist_train_y
                labelsY = self.mnist_label
                '''
                dataX = self.mnistFashion_train_x
                dataY = self.mnistFashion_train_y
                labelsY = self.mnistFashion_label
                '''

            x_train = dataX
            y_train = labelsY

            x_test = np.concatenate((self.mnistFashion_test_x,self.mnist_test_x),axis=0)
            y_test = np.concatenate((self.mnistFashion_test_labels,self.mnistFashion_test_labels),axis=0)

            '''
            input_shape = (28, 28, 1)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            epochs = 20
            batch_size = 64
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            counter = 0
            '''

            n_examples = np.shape(dataX)[0]

            start_epoch = 0
            start_batch_id = 0
            self.num_batches = int(n_examples / self.batch_size)

            # loop for epoch
            start_time = time.time()
            for epoch in range(start_epoch, self.epoch):
                count = 0
                # Random shuffling
                index = [i for i in range(n_examples)]
                random.shuffle(index)
                dataX = dataX[index]
                dataY = dataY[index]
                labelsY = labelsY[index]

                # get batch data
                for idx in range(start_batch_id, self.num_batches):
                    batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_labels = labelsY[idx * self.batch_size:(idx + 1) * self.batch_size]

                    # Train a classifier
                    _, c_class = self.sess.run([self.mini_classifier_optim, self.mini_classLoss],
                                               feed_dict={self.inputs: batch_images, self.y: batch_y,
                                                          self.labels: batch_labels})
                    print(c_class)

                mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x, self.mnistFashion_label)

            #self.saver.save(self.sess, "models/TeacherStudent_MNIST_TO_SVHN")

    def train(self):

        isFirstStage = True
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            #self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion_')

            # saver to save model
            self.saver = tf.train.Saver()

            taskCount = 2

            mnistAccuracy_list = []
            mnistFashionAccuracy_list = []

            for taskIndex in range(taskCount):
                old_Nsamples = 60000
                oldX = self.Generate_GAN_Samples(old_Nsamples, 1)
                oldY = np.zeros((np.shape(oldX)[0], 4))
                oldY[:, 0] = 1

                oldLabels = self.Give_predictedLabels(oldX)
                oldX = oldX[0:np.shape(oldLabels)[0]]
                oldY = oldY[0:np.shape(oldLabels)[0]]

                # define combination of old and new datasets
                # second state
                dataX = np.concatenate((self.mnistFashion_train_x, oldX), axis=0)
                dataY = np.concatenate((self.mnistFashion_train_y, oldY), axis=0)
                dataLabels = np.concatenate((self.mnistFashion_label, oldLabels))
                labelsY = dataLabels

                # third stage
                # dataX = np.concatenate((self.thirdX, oldX), axis=0)
                # dataY = np.concatenate((self.thirdY, oldY), axis=0)

                if taskIndex == 0:
                    isFirstStage = True
                else:
                    isFirstStage = False

                # First stage
                if isFirstStage:
                    dataX = self.mnist_train_x
                    dataY = self.mnist_train_y
                    labelsY = self.mnist_label

                counter = 0

                n_examples = np.shape(dataX)[0]

                start_epoch = 0
                start_batch_id = 0
                self.num_batches = int(n_examples / self.batch_size)

                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_examples)]
                    random.shuffle(index)
                    dataX = dataX[index]
                    dataY = dataY[index]
                    labelsY = labelsY[index]

                    # get batch data
                    for idx in range(start_batch_id, self.num_batches):
                        batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_labels = labelsY[idx * self.batch_size:(idx + 1) * self.batch_size]

                        # update GAN
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                        # update D network
                        _, d_loss = self.sess.run([self.d_optim, self.d_loss],
                                                  feed_dict={self.inputs: batch_images,
                                                             self.z: batch_z, self.y: batch_y})

                        # update G and Q network
                        _, g_loss = self.sess.run(
                            [self.g_optim, self.g_loss],
                            feed_dict={self.inputs: batch_images, self.z: batch_z, self.y: batch_y})

                        # update VAE
                        _, loss1 = self.sess.run([self.vae1_optim, self.vaeLoss],
                                                 feed_dict={self.inputs: batch_images, self.y: batch_y})
                        class_loss = 0

                        # Update VAE by classification loss
                        _, c_class = self.sess.run([self.classifier_optim, self.classifier_loss],
                                                   feed_dict={self.inputs: batch_images, self.y: batch_y})

                        # Train a classifier
                        _, c_class = self.sess.run([self.mini_classifier_optim, self.mini_classLoss],
                                                   feed_dict={self.inputs: batch_images, self.y: batch_y,
                                                              self.labels: batch_labels})

                        outputs1, outputs2 = self.sess.run(
                            [self.output1, self.output2],
                            feed_dict={self.inputs: batch_images, self.y: batch_y})

                        g_outputs = self.sess.run(
                            self.GAN_output,
                            feed_dict={self.inputs: batch_images, self.z: batch_z, self.y: batch_y})

                        # display training status
                        counter += 1
                        print(
                            "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                            % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, loss1, c_class))

                    if isFirstStage == False:
                        mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                        mnistFashionAccuracy = self.Calculate_accuracy(self.mnistFashion_test_x,
                                                                       self.mnistFashion_test_labels)
                        mnistAccuracy_list.append(mnistAccuracy)
                        mnistFashionAccuracy_list.append(mnistFashionAccuracy)
                    else:
                        mnistAccuracy = self.Calculate_accuracy(self.mnist_test_x, self.mnist_label_test)
                        mnistAccuracy_list.append(mnistAccuracy)

            lossArr1 = np.array(mnistAccuracy_list).astype('str')
            f = open("results/MnistToFashion_MNISTAccuracy.txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr1[i])
                f.writelines('\n')
            f.flush()
            f.close()

            lossArr1 = np.array(mnistFashionAccuracy_list).astype('str')
            f = open("results/MnistToFashion_FashionAccuracy.txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr1[i])
                f.writelines('\n')
            f.flush()
            f.close()

            self.saver.save(self.sess, "models/TeacherStudent_MNIST_TO_Fashion")

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
#infoMultiGAN.train_classifier()
#infoMultiGAN.test()
