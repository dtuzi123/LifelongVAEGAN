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

from mnist_hand import *

z_dim = 256

os.environ['CUDA_VISIBLE_DEVICES']='0'

def Code_Classifier(s, scopename, n_hidden, n_output, keep_prob, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

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

    return y1, y

def file_name2(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob(b1)
            t1.append(img_path)

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def file_name(file_dir):
    t1 = []
    file_dir = "../images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../images_background/" + a1 + "/renders/*.png"
            b1 = "../images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)

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

def Generator_Celeba(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def My_Encoder_mnist(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        z_dim = 100
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


def MINI_Classifier(s, scopename, reuse=False):
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

    return y1, y


# Create model of CNN with slim api
def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            # x = tf.reshape(inputs, [-1, 28, 28, 1])
            x = tf.reshape(inputs, [-1, 32, 32, 3])

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
        self.input_height = 64
        self.input_width = 64
        self.c_dim = 3
        self.z_dim = 100
        self.len_discrete_code = 4
        self.epoch = 20

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

    def Predict_Domain(self):
        out_logit, softmaxValue = Encoder_Celeba2_classifier(self.inputs, "classifier", reuse=True)
        label_softmax = tf.nn.softmax(out_logit)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions
    def Predict_Domain_labels(self,testX):
        prediction = self.Predict_Domain()
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = prediction
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions,4)
        return totalPredictions

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
        G1 = Generator_Celeba("GAN_generator", gan_code, reuse=False)

        ## 1. GAN Loss
        # output of D for real images
        D_real_logits = Discriminator_Celeba_WGAN(self.inputs, "discriminator", reuse=False)

        # output of D for fake images
        D_fake_logits = Discriminator_Celeba_WGAN(G1, "discriminator", reuse=True)

        self.g_loss = -tf.reduce_mean(D_fake_logits)
        self.d_loss = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G1
        d_hat = Discriminator_Celeba_WGAN(x_hat, "discriminator", reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx
        self.isPhase = 0

        # domain 1
        z_mean, z_log_sigma_sq = Encoder_Celeba2(self.inputs, "encoder1", batch_size=64, reuse=False)
        out_logit, softmaxValue = Encoder_Celeba2_classifier(self.inputs, "classifier", reuse=False)

        continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

        log_y = tf.log(softmaxValue + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        y_labels = tf.argmax(softmaxValue, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        code1 = tf.concat((continous_variables, discrete_real), axis=1)
        reco1 = Generator_Celeba("generator1", code1, reuse=False)
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
        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "Mini_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def Give_predictedLabels(self, testX):
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

    def Calculate_accuracy(self, testX, testY):
        # testX = self.mnist_test_x
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
            z_mean, z_log_sigma_sq = Encoder_Celeba2(self.inputs, "encoder1", batch_size=64, reuse=True)
            continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

            continous_in = tf.placeholder(tf.float32, [self.batch_size, 256])
            y_in = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
            code1 = tf.concat((continous_in, y_in), axis=1)
            reco1 = Generator_Celeba("generator1", code1, reuse=True)

            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TeacherStudent_Celeba_TO_CACD')

            # load Human face
            batch_size = 64
            img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
            data_files = img_path
            data_files = sorted(data_files)
            data_files = np.array(data_files)  # for tl.iterate.minibatches
            n_examples = 202599
            total_batch = int(n_examples / self.batch_size)

            myIndex = 10
            batch_files = data_files[myIndex * self.batch_size:
                                     (myIndex + 1) * self.batch_size]

            batch = [get_image(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed = batch_images

            myIndex = 11
            batch_files = data_files[myIndex * self.batch_size:
                                     (myIndex + 1) * self.batch_size]

            batch = [get_image(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed_1 = batch_images

            celeba_data_files = data_files

            # load 3D chairs
            # load 3D chairs
            img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
            data_files = img_path
            data_files = sorted(data_files)
            data_files = np.array(data_files)  # for tl.iterate.minibatches
            n_examples = np.shape(data_files)[0]
            total_batch = int(n_examples / self.batch_size)

            batch_files = data_files[0:
                                     self.batch_size]

            # load dataset
            count1 = 0
            image_size = 64

            total_batch = int(n_examples / batch_size)
            total_batch = int(total_batch)
            batch_files = data_files[0:
                                     batch_size]
            batch = [get_image(
                batch_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=False) \
                for batch_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed2 = batch_images

            batch_files = data_files[11 * self.batch_size:
                                     12 * self.batch_size]

            batch = [get_image(
                batch_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=False) \
                for batch_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed2_2 = batch_images

            Chairs_data_files = data_files

            batch_y = np.zeros((batch_size, self.len_discrete_code))
            batch_y[0:32, 0] = 1
            batch_y[32:64, 1] = 1

            x_fixed44 = np.concatenate((x_fixed[0:32], x_fixed2[0:32]), axis=0)

            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            '''
            outputs1, outputs2 = self.sess.run(
                [self.output1, self.output2],
                feed_dict={self.inputs: x_fixed44, self.y: batch_y, self.z: batch_z})
            ims("results/" + "CelebaToCACD_real" + str(0) + ".png", merge2(x_fixed44[:64], [8, 8]))
            ims("results/" + "CelebaToCACD_VAE" + str(0) + ".png", merge2(outputs1[:64], [8, 8]))
            gc = outputs1

            '''

            batchY1 = np.zeros((self.batch_size, self.len_discrete_code))
            batchY1[:, 0] = 1
            batchY2 = np.zeros((self.batch_size, self.len_discrete_code))
            batchY2[:, 1] = 1

            z1 = self.sess.run(
                continous_variables,
                feed_dict={self.inputs: x_fixed2, })
            z2 = self.sess.run(
                continous_variables,
                feed_dict={self.inputs: x_fixed2_2, })

            z_d = z1 - z2
            y_d = batchY1 - batchY2
            y_d = 0
            z_d = z_d / 10.0
            y_d = y_d / 10.0
            myArr = []
            myArr2 = []
            myArr3 = []
            myArr4 = []

            bc = self.Generate_GAN_Samples(128, 2)
            ims("results/" + "CelebaToCADA_GAN" + str(0) + ".png", merge2(bc[0:64], [8, 8]))

            y_new = batchY1
            for index in range(10):
                z_new = z2 + z_d * index
                y_new[:, 0] = 0.1 * index
                xx = self.sess.run(
                    reco1,
                    feed_dict={continous_in: z_new, y_in: y_new})
                myArr.append(xx[0])
                myArr2.append(xx[1])
                myArr3.append(xx[10])
                myArr4.append(xx[11])

            myArr = np.array(myArr)
            myArr2 = np.array(myArr2)
            myArr3 = np.array(myArr3)
            myArr4 = np.array(myArr4)

            ims("results/" + "CelebaToCADA_interpolation" + str(0) + ".png", merge2(myArr, [1, 10]))

            ims("results/" + "CelebaToCharis_interpolation" + str(1) + ".png", merge2(myArr2, [1, 10]))
            ims("results/" + "CelebaToCharis_interpolation" + str(2) + ".png", merge2(myArr3, [1, 10]))
            ims("results/" + "CelebaToCharis_interpolation" + str(3) + ".png", merge2(myArr4, [1, 10]))

            myArr = []
            myArr2 = []
            myArr3 = []
            myArr4 = []
            y_d = batchY2 - batchY1
            y_d = y_d / 10.0
            minZ = -3.0
            maxZ = 3.0
            dx = 6.0 / 10.0

            for myIndex in range(256):
                z1 = self.sess.run(
                    continous_variables,
                    feed_dict={self.inputs: x_fixed, })

                myArr = []
                for index in range(10):
                    z1[0, myIndex] = minZ + dx * index
                    xx = self.sess.run(
                        reco1,
                        feed_dict={continous_in: z1, y_in: batchY1})
                    myArr.append(xx[0])

                myArr = np.array(myArr)
                ims("results/" + "Manipulation" + str(myIndex) + ".png", merge2(myArr, [1, 10]))

            myArr = np.array(myArr)
            ims("results/" + "CelebaToCharis_discrete_interpolation" + str(0) + ".png", merge2(myArr, [1, 10]))

            myIndex = 2

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

    def train(self):

        # load Human face
        batch_size = 64
        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = 202599
        total_batch = int(n_examples / self.batch_size)

        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files

        # load 3D chairs
        img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacdFiles = data_files

        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files

        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        zimuFiles = data_files

        isFirstStage = False
        taskCount = 4
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # saver to save model
            self.saver = tf.train.Saver()

            for taskIndex in range(taskCount):

                if taskIndex != 0:
                    # load generated images
                    old_Nsamples = 50000
                    oldX = self.Generate_GAN_Samples(old_Nsamples, 1)
                    oldY = self.Predict_Domain_labels(oldX)
                    oldX = oldX[0:np.shape(oldY)[0]]
                    oldX_number = int(old_Nsamples / 32)

                if taskIndex == 0:
                    x_train = celebaFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 1:
                    x_train = cacdFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 2:
                    x_train = chairFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 3:
                    x_train = zimuFiles
                    x_fixed = x_train[0:batch_size]

                n_samples = np.shape(np.array(x_train))[0]
                total_batch = int(n_samples / batch_size)

                start_epoch = 0
                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_samples)]
                    random.shuffle(index)
                    x_train = x_train[index]

                    counter = 0
                    # get batch data
                    for i in range(total_batch):

                        batchFiles = x_train[i * batch_size:i * batch_size + batch_size]

                        if taskIndex == 0:
                            batch = [get_image(
                                sample_file,
                                input_height=128,
                                input_width=128,
                                resize_height=64,
                                resize_width=64,
                                crop=True)
                                for sample_file in batchFiles]
                        elif taskIndex == 1:
                            batch = [get_image(
                                sample_file,
                                input_height=250,
                                input_width=250,
                                resize_height=64,
                                resize_width=64,
                                crop=True)
                                for sample_file in batchFiles]
                        elif taskIndex == 2:
                            image_size = 64
                            batch = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                                     for batch_file in batchFiles]
                        elif taskIndex == 3:
                            batch = [get_image(batch_file, 105, 105,
                                               resize_height=64, resize_width=64,
                                               crop=False, grayscale=False) \
                                     for batch_file in batchFiles]
                            batch = np.array(batch)
                            batch = np.reshape(batch, (64, 64, 64, 1))
                            batch = np.concatenate((batch, batch, batch), axis=-1)

                        # Compute the offset of the current minibatch in the data.
                        batch_xs_target = batch
                        x_fixed = batch
                        batch_xs_input = batch
                        batch_images = batch
                        batch_images = np.array(batch_images)

                        if taskIndex !=0:
                            count = int(np.shape(oldX)[0]/self.batch_size)
                            newIndex = i % count
                            newBatch = oldX[newIndex*self.batch_size:(newIndex+1)*self.batch_size]
                            newY = oldY[newIndex*self.batch_size:(newIndex+1)*self.batch_size]

                            currentY = np.zeros((self.batch_size,4))
                            currentY[:,taskIndex] = 1
                            batch_images = np.concatenate((batch_images,newBatch),axis=0)
                            batch_y = np.concatenate((currentY,newY))
                            index = [i for i in range(self.batch_size*2)]
                            random.shuffle(index)
                            batch_images = batch_images[index]
                            batch_images = batch_images[0:self.batch_size]
                            batch_y = batch_y[index]
                            batch_y = batch_y[0:self.batch_size]
                        else:
                            batch_y = np.zeros((self.batch_size,4))
                            batch_y[:,taskIndex] = 1

                        # update GAN
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                        ims("results/" + "real" + str(3) + ".png", merge2(batch_images[:64], [8, 8]))

                        outputs1, outputs2 = self.sess.run(
                            [self.output1, self.output2],
                            feed_dict={self.inputs: batch_images, self.y: batch_y, self.z: batch_z})
                        ims("results/" + "VAE" + str(0) + ".jpg", merge2(outputs1[:64], [8, 8]))
                        gc = outputs1

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
                                                 feed_dict={self.inputs: batch_images, self.y: batch_y, self.z: batch_z})
                        class_loss = 0


                        # Update VAE by classification loss
                        _, c_class = self.sess.run([self.classifier_optim, self.classifier_loss],
                                                   feed_dict={self.inputs: batch_images, self.y: batch_y, self.z: batch_z})

                        # display training status
                        counter += 1
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                              % (epoch, i, 0, time.time() - start_time, d_loss, g_loss, loss1, c_class))

                    outputs1, outputs2 = self.sess.run(
                        [self.output1, self.output2],
                        feed_dict={self.inputs: batch_images, self.y: batch_y, self.z: batch_z})

                    g_outputs = self.sess.run(
                        self.GAN_output,
                        feed_dict={self.inputs: batch_images, self.z: batch_z, self.y: batch_y})

                    y_RPR1 = np.reshape(outputs1, (-1, 64, 64, 3))
                    g_output = np.reshape(g_outputs, (-1, 64, 64, 3))
                    ims("results/" + "Celeba" + str(epoch) + ".jpg", merge2(y_RPR1[:64], [8, 8]))
                    ims("results/" + "GAN" + str(epoch) + ".jpg", merge2(g_output[:64], [8, 8]))

                self.saver.save(self.sess, "models/TeacherStudent_Celeba_Long")


infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
#infoMultiGAN.test()
