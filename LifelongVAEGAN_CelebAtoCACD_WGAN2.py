#from keras.datasets import mnist
import time
from utils import *
#from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
import tensorflow.keras as keras
from glob import glob
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
'''
from Basic_structure import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow as tf2
from cv2_imageProcess import *

os.environ['CUDA_VISIBLE_DEVICES']='3'
#
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

def My_Encoder_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
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

def My_Classifier_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        # z_dim = 32
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
    with tf.compat.v1.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(s, 400, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 2
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')

        return logoutput

# Create model of CNN with slim api
def Image_classifier(inputs, scopename, is_training=True, reuse=False):
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

def CodeImage_classifier(s, scopename, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

        # initializers
        w_init = tf.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        n_hidden = 500
        keep_prob = 0.9

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 4
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y

import Fid_tf2 as fid2

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 64
        self.input_width = 64
        self.c_dim = 3
        self.z_dim = 256
        self.len_discrete_code = 4
        self.epoch = 50
        #
        self.learning_rate = 1e-4
        self.beta1 = 0.5

        self.GenerationArr = []
        self.DLossArr = []
        self.GLossArr = []
        self.FixedMemory = []
        self.totalTrainingSet = []
        self.totalTestingSet = []

        self.beta = 0.02

        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files
        maxSize = 50000
        celebaTraining = celebaFiles[0:maxSize]
        celebaTesting = celebaFiles[maxSize:maxSize + 5000]

        '''
        batch = [GetImage_cv_255(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaTraining[0:self.batch_size]]

        print(np.min(batch))
        print(np.max(batch))
        '''

    def Return_Reconstructions(self,x):
        count = int(np.shape(x)[0] / self.batch_size)
        arr = []
        for i in range(count):
            batch = x[i*self.batch_size:(i+1)*self.batch_size]
            reco = self.sess.run(self.Reco,feed_dict={self.inputs:batch})
            if np.shape(arr)[0] == 0:
                arr = reco
            else:
                arr = np.concatenate((arr,reco),axis=0)
        return arr

    def Give_reconstructions(self):
        z_mean1, z_log_sigma_sq1 = Encoder_Celeba2(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain",reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_Celeba("VAE_Generator", code, reuse=True)
        return VAE1

    def Create_subloss(self,G):
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G
        d_hat = Discriminator_Celeba(x_hat, "discriminator", reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        #self.d_loss = self.d_loss + ddx
        return ddx

    def Select_Expert(self,index):
        expertIndex = index + 1
        generatorStr = "GAN_generator" + str(expertIndex)

        gan_code = self.z
        G1 = Generator_Celeba(generatorStr, gan_code, reuse=True)
        self.GenerationArr.append(G1)

        ## 1. GAN Loss
        # output of D for real images
        D_real_logits = Discriminator_Celeba(self.inputs, "discriminator", reuse=True)

        # output of D for fake images
        D_fake_logits1 = Discriminator_Celeba(G1, "discriminator", reuse=True)

        g_loss1 = -tf.reduce_mean(D_fake_logits1)

        d_loss1 = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits1)

        d_loss1 = self.d_loss1 + self.Create_subloss(G1)
        #self.DLossArr.append(d_loss1)
        #self.GLossArr.append(g_loss1)

        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith(generatorStr)]

        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in discriminator_vars1]

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.d_optim1 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(d_loss1, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(g_loss1, var_list=GAN_generator_vars1)

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        self.sess.run(tf.variables_initializer(not_initialized_vars))

    def Build_NewExerpt(self):
        NewIndex = self.componentCount + 1
        generatorStr = "GAN_generator" + str(NewIndex)
        discriminatorStr = "discriminator" + str(NewIndex)
        gan_code = self.z
        G1 = Generator_Celeba_Tanh(generatorStr, gan_code, reuse=False)
        self.GenerationArr.append(G1)

        ## 1. GAN Loss
        # output of D for real images
        _, D_real_logits = Discriminator_Celeba_Orginal(self.inputs, discriminatorStr, reuse=False)

        # output of D for fake images
        _, D_fake_logits = Discriminator_Celeba_Orginal(G1, discriminatorStr, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(D_real_logits, tf.ones_like(D_real_logits)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(D_fake_logits, tf.zeros_like(D_fake_logits)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(D_fake_logits, tf.ones_like(D_fake_logits)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.DLossArr.append(self.d_loss)
        self.GLossArr.append(self.g_loss)

        self.label_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=predict2, labels=self.d))


        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith(discriminatorStr)]
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith(generatorStr)]
        infer2 = [var for var in T_vars if var.name.startswith(generatorStr)]

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.d_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.d_loss, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.g_loss, var_list=GAN_generator_vars1)
            self.vae_optim = tf.train.AdamOptimizer(learning_rate=0.0001) \
                .minimize(vaeLoss, var_list=VAE_parameters)
            self.classifier2 = tf.train.AdamOptimizer(learning_rate=0.0001) \
                .minimize(self.label_loss2, var_list=infer2)

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        self.sess.run(tf.variables_initializer(not_initialized_vars))


    def sample_gumbel(self,shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax(self,logits, temperature, hard=False):
        gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)

        if hard:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                             y.dtype)
            y = tf.stop_gradient(y_hard - y) + y

        return y

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.d = tf.placeholder(tf.float32, [self.batch_size, 2])

        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.weights = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.index = tf.placeholder(tf.int32, [self.batch_size])
        self.gan_inputs = tf.placeholder(tf.float32, [bs] + image_dims)
        self.gan_domain = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.gan_domain_labels = tf.placeholder(tf.float32, [self.batch_size, 1])

        inferenceName = "Inference"
        inferenceName2 = "Inference2"

        # GAN networks
        gan_code = self.z
        G1 = Generator_Celeba_Tanh("GAN_generator1", gan_code,self.batch_size, reuse=False)
        self.GenerationArr.append(G1)

        z_mean, z_log_sigma_sq = Encoder_Celeba2(self.inputs, inferenceName,self.batch_size, reuse=False)

        code = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1,
                                                          dtype=tf.float32)

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_log_sigma_sq) - tf.log(1e-8 + tf.square(z_log_sigma_sq)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        predict2 = MINI_Classifier(code, inferenceName2, reuse=False)
        p2 = self.gumbel_softmax(predict2,0.5)
        #newCode = tf.concat([code,p2],1)
        newCode = code

        reco = Generator_Celeba_Tanh("GAN_generator1", newCode,self.batch_size, reuse=True)
        self.Reco = reco

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco - self.inputs), [1, 2, 3]))

        p2_q_y = tf.nn.softmax(p2)
        p2_log_q_y = tf.log(p2_q_y + 1e-20)

        kl_tmp2 = tf.reshape(p2_q_y * (p2_log_q_y - tf.log(1.0 / 2)),
                             [-1, 64, 2])
        kl_tmp2 = tf.reduce_sum(kl_tmp2, [1, 2])

        self.vaeLoss = reconstruction_loss1 + KL_divergence1#+kl_tmp2

        ## 1. GAN Loss
        # output of D for real images
        _, D_real_logits = Discriminator_Celeba_Orginal(self.inputs, "discriminator",self.batch_size, reuse=False)

        # output of D for fake images
        _, D_fake_logits = Discriminator_Celeba_Orginal(G1,"discriminator",self.batch_size, reuse=True)

        self.g_samples = G1

        self.g_loss1 = -tf.reduce_mean(D_fake_logits)  # + vaeLoss * mybeta
        self.d_loss1 = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G1
        d_hat = Discriminator_Celeba_Orginal(x_hat, "discriminator",self.batch_size, reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss1 = self.d_loss1 + ddx

        self.d_loss = self.d_loss1
        self.g_loss = self.g_loss1
        self.DLossArr.append(self.d_loss)
        self.GLossArr.append(self.g_loss)

        self.GAN_gen1 = G1
        self.isPhase = 0

        self.label_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=predict2, labels=self.d))

        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith('GAN_generator1')]
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith(inferenceName)]

        infer2 = [var for var in T_vars if var.name.startswith(inferenceName2)]

        vae_vars = VAE_encoder_vars  + infer2 + GAN_generator_vars1

        # optimizers
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):

            self.d_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.d_loss, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.g_loss, var_list=GAN_generator_vars1)

            '''
            self.d_optim1 = tf.train.AdamOptimizer(self.learning_rate,beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.AdamOptimizer(self.learning_rate,beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=GAN_generator_vars1 + VAE_encoder_vars)
            '''

            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
            self.classifier2 = tf.train.AdamOptimizer(learning_rate=0.0001) \
                .minimize(self.label_loss2, var_list=infer2)


        self.fakeImage = G1
        b1 = 0

    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[64, 1, 1, 1], minval=0., maxval=1.)
        differences = self.g_samples - self.inputs
        interpolates = self.inputs + (alpha * differences)
        gradients = tf.gradients(Discriminator_Celeba_Orginal(interpolates,"discriminator",self.batch_size, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        return gradient_penalty

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
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

    def Make_interplation(self,test1,test2,index=0):
        z_mean1, z_log_sigma_sq1 = Encoder_Celeba2(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain",reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        z_input = tf.placeholder(tf.float32, [64, 256])
        d_input = tf.placeholder(tf.float32, [64, 4])
        code = tf.concat((z_input, d_input), axis=1)
        VAE1 = Generator_Celeba_Tanh("VAE_Generator", code, reuse=True)

        z1,d1 = self.sess.run([continous_variables1,discrete_real],feed_dict={self.inputs:test1})
        z2,d2 = self.sess.run([continous_variables1,discrete_real],feed_dict={self.inputs:test2})

        z_dis = z2 - z1
        d_dis = d2 - d1
        z_dis = z_dis / 12.0
        d_dis = d_dis / 12.0

        arr1 = []
        for i in range(12):
            z = z1 + z_dis*i
            d = d1 + d_dis*i
            reco = self.sess.run(VAE1,feed_dict={z_input:z,d_input:d})
            arr1.append(reco[index])

        arr1 = np.array(arr1)
        return arr1

    def Student_Generation(self):
        z_input = tf.placeholder(tf.float32, [64, 256])
        d_input = tf.placeholder(tf.float32, [64, 4])
        code = tf.concat((z_input, d_input), axis=1)
        VAE1 = Generator_Celeba_Tanh("VAE_Generator", code, reuse=True)

        for i in range(4):
            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
            batch_d = np.zeros((self.batch_size,4))
            batch_d[:,i] = 1
            generation = self.sess.run(VAE1,feed_dict={z_input:batch_z,d_input:batch_d})
            ims("results/" + "StudentGeneration_" + str(i) + ".png", merge2(generation[:25], [5, 5]))


    def LoadDataStream(self):
        image_size = 64

        img_path = glob('../img_celeba2/*.jpg')
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files
        celebaTraining = celebaTraining[0:50000]
        celebaTesting = celebaTesting[50000:5500]

        celebaTraining = [get_image(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaTraining]

        celebaTesting = [get_image(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaTesting]

        img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacd = data_files
        cacdTraining = cacd[0:50000]
        cacdTesting = cacd[50000:55000]

        cacdTraining = [get_image(
            sample_file,
            input_height=250,
            input_width=250,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in cacdTraining]

        cacdTesting = [get_image(
            sample_file,
            input_height=250,
            input_width=250,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in cacdTesting]

        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files
        chairTraining = chairFiles[0:50000]
        chairTesting = chairFiles[50000:55000]

        chairTraining = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                 for batch_file in chairTraining]

        chairTesting = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                         for batch_file in chairTesting]

        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        zimuFiles = data_files
        zimuTraining = zimuFiles[0:50000]
        zimuTesting = zimuFiles[50000:55000]

        zimuTraining = [get_image(batch_file, 105, 105,
                           resize_height=64, resize_width=64,
                           crop=False, grayscale=False) \
                 for batch_file in zimuTraining]
        zimuTraining = np.array(zimuTraining)
        zimuTraining = np.reshape(zimuTraining, (-1, 64, 64, 1))
        zimuTraining = np.concatenate((zimuTraining, zimuTraining, zimuTraining), axis=-1)

        zimuTesting = [get_image(batch_file, 105, 105,
                                  resize_height=64, resize_width=64,
                                  crop=False, grayscale=False) \
                        for batch_file in zimuTesting]
        zimuTesting = np.array(zimuTesting)
        zimuTesting = np.reshape(zimuTesting, (-1, 64, 64, 1))
        zimuTesting = np.concatenate((zimuTesting, zimuTesting, zimuTesting), axis=-1)

        self.totalTrainingSet = np.concatenate((celebaTraining,cacdTraining,chairTraining,zimuTraining),axis=0)
        self.totalTestingSet = np.concatenate((celebaTesting,cacdTesting,chairTesting,zimuTesting),axis=0)

    def GiveInteplationResults(self,x1,x2):
        z1 = self.sess.run(self.Student_z,feed_dict={self.inputs:x1,self.gan_inputs:x1})
        z2 = self.sess.run(self.Student_z,feed_dict={self.inputs:x2,self.gan_inputs:x2})

        totalCount = 12
        zz = (z1 - z2) / 12.0
        newX = []
        for i in range(totalCount):
            newz = z2 + i * zz
            xx = self.sess.run(self.studentGAN,feed_dict={self.z:newz})
            newX.append(xx[0])
        newX = np.array(newX)
        return newX


    def GiveInteplationResults_Inverse(self,x1,x2):
        z1 = self.sess.run(self.Student_z,feed_dict={self.inputs:x1,self.gan_inputs:x1})
        z2 = self.sess.run(self.Student_z,feed_dict={self.inputs:x2,self.gan_inputs:x2})

        totalCount = 12
        zz = (z2 - z1) / 12.0
        newX = []
        for i in range(totalCount):
            newz = z1 + i * zz
            xx = self.sess.run(self.studentGAN,feed_dict={self.z:newz})
            newX.append(xx[0])
        newX = np.array(newX)
        return newX


    def test(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'model/GANTeacher64_twotask_LTS')

            self.LoadTestingData()

            recoloss1, fid1 = self.Calculate_RecoAndFid_ByDataset(self.celebaData)
            # recoloss2, fid2 = self.Calculate_RecoAndFid_ByDataset(self.cacdData)
            recoloss3, fid3 = self.Calculate_RecoAndFid_ByDataset(self.chairData)
            # recoloss4, fid4 = self.Calculate_RecoAndFid_ByDataset(self.zimuData)

            batch1 = self.celebaData[0:self.batch_size]
            batch2 = self.chairData[0:self.batch_size]

            reco1 = self.sess.run(self.Reco, feed_dict={self.inputs: batch1, self.gan_inputs: batch1})
            reco2 = self.sess.run(self.Reco, feed_dict={self.inputs: batch2, self.gan_inputs: batch2})
            batch1 = np.array(batch1)
            batch2 = np.array(batch2)
            ims_cv("results/" + "LTS_final64_real1" + ".png", merge2(batch1[:64], [8, 8]))
            ims_cv("results/" + "LTS_final64_reco1" + ".png", merge2(reco1[:64], [8, 8]))
            ims_cv("results/" + "LTS_final64_real2" + ".png", merge2(batch2[:64], [8, 8]))
            ims_cv("results/" + "LTS_final64_reco2" + ".png", merge2(reco2[:64], [8, 8]))

            print("Reco Loss")
            print(recoloss1)
            # print(recoloss2)
            print(recoloss3)
            # print(recoloss4)
            # sum1 = recoloss1 + recoloss2 + recoloss3 + recoloss4
            sum1 = recoloss1 + recoloss3
            sum1 = sum1 / 2
            print(sum1)

            print("FID score")
            print(fid1)
            # print(fid2)
            print(fid3)
            # print(fid4)
            # sum2 = fid1 + fid2 + fid3 + fid4
            sum2 = fid1 + fid3
            sum2 = sum2 / 2
            print(sum2)


    def SaveImage2(self, str, image):
        newGan = (image + 1) * 127.5
        newGan = np.reshape(newGan, (-1, 64, 64, 3))
        str2 = str + '.png'
        cv2.imwrite(os.path.join("results/", str2),
                    merge2(newGan[:12], [1, 12]))


    def Generate_GAN_Samples(self, n_samples, classN):
        myArr = []
        for tt in range(classN):
            y1 = np.zeros((self.batch_size, 4))
            y1[:, 0] = 1
            num1 = int(n_samples / self.batch_size)
            for i in range(num1):
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                batch_labels = np.random.multinomial(1,
                                                     10 * [
                                                         float(1.0 / 10)],
                                                     size=[self.batch_size])
                g_outputs = self.sess.run(
                    self.GAN_output,
                    feed_dict={self.z: batch_z, self.y: y1, self.labels: batch_labels})
                for t1 in range(self.batch_size):
                    myArr.append(g_outputs[t1])

        myArr = np.array(myArr)
        return myArr

    def SelectExpert(self,testX,weights,domainState, taskIndex):
        gan_code = self.z

        batch_labels = np.random.multinomial(1,
                                             self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                             size=[self.batch_size])
        # update GAN
        batch_z = np.random.normal(size=[self.batch_size, self.z_dim])

        G1 = Generator_Celeba_Tanh("GAN_generator1", gan_code, reuse=True)
        G2 = Generator_Celeba_Tanh("GAN_generator2", gan_code, reuse=True)
        G3 = Generator_Celeba_Tanh("GAN_generator3", gan_code, reuse=True)
        G4 = Generator_Celeba_Tanh("GAN_generator4", gan_code, reuse=True)

        D_real_logits = Discriminator_Celeba(self.inputs, "discriminator", reuse=True)
        # output of D for fake images
        D_fake_logits1 = Discriminator_Celeba(G1, "discriminator", reuse=True)
        D_fake_logits2 = Discriminator_Celeba(G2, "discriminator", reuse=True)
        D_fake_logits3 = Discriminator_Celeba(G3, "discriminator", reuse=True)
        D_fake_logits4 = Discriminator_Celeba(G4, "discriminator", reuse=True)

        d_loss1 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits1))
        d_loss2 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits2))
        d_loss3 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits3))
        d_loss4 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits4))

        d_loss1_, d_loss2_, d_loss3_, d_loss4_ = self.sess.run([d_loss1, d_loss2, d_loss3, d_loss4],
                                                               feed_dict={self.inputs: testX,
                                                                          self.z: batch_z,
                                                                          })
        d_loss1_ = d_loss1_ + (1 - weights[0, 0]) * 10000
        d_loss2_ = d_loss2_ + (1 - weights[0, 1]) * 10000
        d_loss3_ = d_loss3_ + (1 - weights[0, 2]) * 10000
        d_loss4_ = d_loss4_ + (1 - weights[0, 3]) * 10000

        score = []
        score.append(d_loss1_)
        score.append(d_loss2_)
        score.append(d_loss3_)
        score.append(d_loss4_)
        index = score.index(min(score))
        if index == 0:
            weights[:, 0] = 0
            tmp = domainState[0]
            domainState[0] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 1:
            weights[:, 1] = 0
            # tmp = domainState[1]
            domainState[1] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 2:
            weights[:, 2] = 0
            # tmp = domainState[2]
            domainState[2] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 3:
            weights[:, 3] = 0
            # tmp = domainState[3]
            domainState[3] = taskIndex
            # domainState[taskIndex] = tmp

        return weights, domainState

    def SelectGANs_byIndex(self,index):
        if index == 1:
            return self.GAN_gen1
        elif index == 2:
            return self.GAN_gen2
        elif index == 3:
            return self.GAN_gen3
        elif index ==4:
            return self.GAN_gen4

    def GenerateSamplesBySelect(self,n,index):
        myGANs = self.GenerationArr[index-1]
        a = int(n/self.batch_size)
        myArr = []
        for i in range(a):
            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
            aa = self.sess.run(myGANs, feed_dict={self.z: batch_z})
            for t in range(self.batch_size):
                myArr.append(aa[t])
        myArr = np.array(myArr)
        return myArr

    def Calculate_FID_Score(self,nextTaskIndex):
        fid2.session = self.sess

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

        batch_size = self.batch_size
        if nextTaskIndex == 0:
            x_train = celebaFiles
            x_fixed = x_train[0:batch_size]
        elif nextTaskIndex == 1:
            x_train = cacdFiles
            x_fixed = x_train[0:batch_size]
        elif nextTaskIndex == 2:
            x_train = chairFiles
            x_fixed = x_train[0:batch_size]
        elif nextTaskIndex == 3:
            x_train = zimuFiles
            x_fixed = x_train[0:batch_size]

        batchFiles = x_train[0:1000]

        if nextTaskIndex == 0:
            '''
            batch = [get_image(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batchFiles]
            '''
            batch = [GetImage_cv_255(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batchFiles]
        elif nextTaskIndex == 1:
            '''
            batch = [get_image(
                sample_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batchFiles]
            '''
            batch = [GetImage_cv_255(
                sample_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batchFiles]
        elif nextTaskIndex == 2:
            image_size = 64
            '''
            batch = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                     for batch_file in batchFiles]
            '''
            batch = [GetImage_cv2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                     for batch_file in batchFiles]
        elif nextTaskIndex == 3:
            '''
            batch = [get_image(batch_file, 105, 105,
                               resize_height=64, resize_width=64,
                               crop=False, grayscale=False) \
                     for batch_file in batchFiles]
            '''
            batch = [GetImage_cv2_255(batch_file,105,False,
                               64) \
                     for batch_file in batchFiles]
            batch = np.array(batch)
            batch = np.reshape(batch, (-1, 64, 64, 1))
            batch = np.concatenate((batch, batch, batch), axis=-1)

        myCount = 1000
        realImages = batch[0:myCount]
        realImages = np.transpose(realImages, (0, 3, 1, 2))
        realImages = ((realImages + 1.0) * 255) / 2.0

        #Calculate FID
        fidArr = []
        for tIndex in range(self.componentCount):
            myIndex = tIndex + 1
            fakeImages = []
            myGANs = self.GenerationArr[tIndex]
            tt = int(myCount / self.batch_size)
            for i in range(tt):
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                aa = self.sess.run(myGANs, feed_dict={self.z: batch_z})
                if i == 0:
                    bachFake = aa
                for j in range(self.batch_size):
                    fakeImages.append(aa[j])

            fakeImages = np.array(fakeImages)
            realImages = realImages[0:np.shape(fakeImages)[0]]
            fakeImages = np.transpose(fakeImages, (0, 3, 1, 2))
            fakeImages = ((fakeImages + 1.0) * 255) / 2.0

            fidScore = fid2.get_fid(realImages, fakeImages)
            print(fidScore)
            fidArr.append(fidScore)

        #Compare FID
        minIndex = fidArr.index(min(fidArr))
        minFid = min(fidArr)
        minIndex = minIndex + 1

        return minIndex,minFid

    def MakePseudoData(self):

        dataList = []
        labelList = []
        batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
        x0 = self.sess.run(self.GenerationArr[0], feed_dict={self.z: batch_z})

        batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
        x1 = self.sess.run(self.GenerationArr[0], feed_dict={self.z: batch_z})

        return x1

    def Calculate_FID_ByAllLeanredComponents(self):
        totalSet = []

        memoryCount = np.shape(self.FixedMemory)[0]
        memoryCount = int(memoryCount / self.batch_size)
        dataCount = memoryCount
        for kk in range(self.componentCount - 1):
            arr1 = []
            for k1 in range(dataCount):
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])

                ComponentIndex = kk
                generatedImages = self.sess.run(self.GenerationArr[ComponentIndex], feed_dict={self.z: batch_z})
                generatedImagesB = generatedImages
                if np.shape(arr1)[0] == 0:
                    arr1 = generatedImagesB
                else:
                    arr1 = np.concatenate((arr1, generatedImagesB), axis=0)

            arr1 = np.array(arr1)
            totalSet.append(arr1)

        memory = self.FixedMemory[0: dataCount * self.batch_size]

        fidArr = []
        for i in range(self.componentCount - 1):
            mse1 = self.FID_Evaluation(memory, totalSet[i])
            fidArr.append(mse1)

        return fidArr

    def FID_Evaluation(self, recoArr, test):

        recoArr = np.array(recoArr)
        test = np.array(test)

        recoArr = np.reshape(recoArr, (-1, 32, 32, 3))
        test = np.reshape(test, (-1, 32, 32, 3))

        fid2.session = self.sess

        test1 = np.transpose(test, (0, 3, 1, 2))
        # test1 = ((realImages + 1.0) * 255) / 2.0
        #test1 = test1 * 255.0
        test1 = (test1 + 1) * 127.5

        test2 = np.transpose(recoArr, (0, 3, 1, 2))
        # test1 = ((realImages + 1.0) * 255) / 2.0
        #test2 = test2 * 255.0
        test2 = (test2 + 1) * 127.5

        fidScore = fid2.get_fid(test1, test2)
        return fidScore

    def Calculate_RecoAndFid_ByDataset(self,dataset):
        count = int(np.shape(dataset)[0] / self.batch_size)
        sum1 = 0
        recoArr = []
        realArr = []
        for i in range(count):
            batch1 = dataset[i*self.batch_size:(i+1)*self.batch_size]
            recoLoss,reco = self.sess.run([self.vaeLoss,self.Reco],
                feed_dict={self.inputs: batch1,self.gan_inputs:batch1})
            sum1 = sum1 + recoLoss

            if np.shape(realArr)[0] == 0:
                realArr = batch1
            else:
                realArr = np.concatenate((realArr,batch1),axis=0)

            if np.shape(recoArr)[0] == 0:
                recoArr = reco
            else:
                recoArr = np.concatenate((recoArr,reco),axis=0)

        sum1 = sum1 / count
        fidScore = self.FID_Evaluation(realArr, recoArr)
        return sum1,fidScore

    def LoadTestingData(self):
        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files
        maxTestingSize = 5000
        maxSize = 50000
        celebaTraining = celebaFiles[0:np.shape(celebaFiles)[0] - maxTestingSize]
        celebaTesting = celebaFiles[maxSize:maxSize+maxSize]
        image_size = 64

        celebaTesting = [GetImage_cv(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in celebaTesting]

        self.celebaData = celebaTesting

        img_path = glob('../CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacd = data_files
        cacdTraining = cacd[0:50000]
        cacdTesting = cacd[50000:55000]

        cacdTesting = [GetImage_cv(
                sample_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in cacdTesting]

        self.cacdData = cacdTesting

        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files
        chairTraining = chairFiles[0:50000]
        chairTesting = chairFiles[50000:55000]


        chairTesting = [GetImage_cv2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                     for batch_file in chairTesting]

        self.chairData = chairTesting

        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        zimuFiles = data_files
        maxTestingSize = 5000
        zimuTraining = zimuFiles[0:np.shape(zimuFiles)[0] - maxTestingSize]
        zimuTesting = zimuFiles[np.shape(zimuFiles)[0] - maxTestingSize : np.shape(zimuFiles)[0]]

        zimuTesting = [GetImage_cv_255_Low(batch_file, 105, 105, 64, 64, True
                                         ) \
                     for batch_file in zimuTesting]
        zimuTesting = np.array(zimuTesting)
        zimuTesting = np.reshape(zimuTesting, (-1, 64, 64, 1))
        zimuTesting = np.concatenate((zimuTesting, zimuTesting, zimuTesting), axis=-1)

        self.zimuData = zimuTesting

    def GenerateByCount(self,n):
        count = int(n/self.batch_size)
        arr = []
        for i in range(count):
            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
            gen = self.sess.run(self.GenerationArr[0],feed_dict={self.z:batch_z})
            for j in range(self.batch_size):
                arr.append(gen[j])
        arr = np.array(arr)
        return arr


    def train(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        image_size = 64

        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files
        maxSize = 50000
        maxTestingSize = 5000
        celebaTraining = celebaFiles[0:maxSize]
        celebaTesting = celebaFiles[maxSize: maxSize + maxTestingSize]

        celebaTrainSet = [GetImage_cv(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaTraining]
        celebaTrainSet = np.array(celebaTrainSet)

        celebaTesting = [GetImage_cv(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaTesting]
        celebaTesting = np.array(celebaTesting)
        self.celebaTesting = celebaTesting

        # ims_cv("results/bbbssss.png",merge2(celebaTrainSet[:64], [8, 8]))

        # img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
        img_path = glob('../CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacd = data_files
        cacdTraining = cacd[0:50000]
        cacdTesting = cacd[50000:55000]

        cacdTrainSet = [GetImage_cv_255(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in cacdTraining[0:64]]
        cacdTrainSet = np.array(cacdTrainSet)

        print(np.shape(cacdTraining))
        batch = [GetImage_cv_255(
            sample_file,
            input_height=250,
            input_width=250,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in cacdTesting]
        batch = np.array(batch)
        cacdTesting = batch
        ims_cv("results/dddd.png", merge2(batch[:64], [8, 8]))


        '''
        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files

        aa = 1000
        chairTraining = chairFiles[0:maxSize]
        chairTesting = chairFiles[maxSize:maxSize + 5000]

        batch = [GetImage_cv2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                 for batch_file in chairTraining]
        batch = np.array(batch)
        chairTraining = batch

        batch = [GetImage_cv2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                 for batch_file in chairTesting]
        batch = np.array(batch)
        chairTesting = batch
        self.chairTesting = chairTesting

        # ims_cv("results/aaa.png",merge2(batch[:64], [8, 8]))
        '''

        self.totalSet = []
        self.totalSet.append(celebaTrainSet)
        self.totalSet.append(cacdTraining)

        #self.totalSet = []
        #self.totalSet.append(celebaTrainSet)
        #self.totalSet.append(chairTraining)

        taskCount = 4

        self.componentCount = 1
        self.currentComponent = 1
        self.fid_hold = 200
        self.IsAdd = 0

        #
        self.maxMmeorySize = 2500

        self.STM_Memory = []

        isFirstStage = False
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # saver to save model
            self.saver = tf.train.Saver()
            batch_size = self.batch_size
            ExpertWeights = np.ones((self.batch_size,4))

            DomainState = np.zeros(4).astype(np.int32)
            DomainState[0] = 0
            DomainState[1] = 1
            DomainState[2] = 2
            DomainState[3] = 3

            self.checkCount = 0
            #
            taskCount = 2
            currentStep = 0
            taskCount = 2
            #
            for kk in range(taskCount):
                currentSet = self.totalSet[kk]
                currentSetTask = np.zeros((np.shape(currentSet)[0],2))
                currentSetTask[:,kk] = 1

                print("task")
                print(np.shape(currentSet))

                if kk > 0:
                    generated = self.GenerateByCount(50000)
                    generatedTask = np.zeros((np.shape(generated)[0],2))
                    generatedTask[:,0] = 1

                    currentSet = np.concatenate((currentSet,generated),0)
                    currentSetTask = np.concatenate((currentSetTask,generatedTask),0)

                    #currentSet = np.concatenate((currentSet,generated),axis=0)

                    trainingCount = int(np.shape(currentSet)[0]/self.batch_size)
                    for m in range(self.epoch):
                        n_examples = np.shape(currentSet)[0]
                        index2 = [i for i in range(n_examples)]
                        np.random.shuffle(index2)
                        currentSet = currentSet[index2]
                        currentSetTask = currentSetTask[index2]
                        smallBatch = 64

                        for j in range(trainingCount):

                            #batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])

                            batchImage = currentSet[j * smallBatch:(j + 1) * smallBatch]
                            batchTask = currentSetTask[j * smallBatch:(j + 1) * smallBatch]

                            if j % 5 == 0:
                                # update G and Q network
                                _, g_loss = self.sess.run(
                                    [self.g_optim1, self.g_loss],
                                    feed_dict={self.inputs: batchImage, self.z: batch_z, self.d: batchTask})

                            # update D network
                            _, d_loss = self.sess.run([self.d_optim1, self.d_loss],
                                                      feed_dict={self.inputs: batchImage,
                                                                 self.z: batch_z, self.d: batchTask})

                            # Training student
                            _ = self.sess.run(
                                self.vae_optim,
                                feed_dict={self.inputs: batchImage, self.z: batch_z, self.gan_inputs: batchImage,self.d:batchTask})

                            # Training student
                            _ = self.sess.run(
                                self.classifier2,
                                feed_dict={self.inputs: batchImage, self.z: batch_z, self.gan_inputs: batchImage,self.d:batchTask})

                        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                              .format(0, 0, 0, 0, g_loss, d_loss, self.componentCount, ))

                        print("save the image")
                        batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                        gan1 = self.sess.run(self.GenerationArr[np.shape(self.GenerationArr)[0] - 1],
                                             feed_dict={self.z: batch_z})
                        ims_cv("results/" + "myTest_" + str(m) + "_" + str(kk) + ".png", merge2(gan1[:64], [8, 8]))

                else:
                    for m in range(self.epoch):
                        n_examples = np.shape(currentSet)[0]
                        index2 = [i for i in range(n_examples)]
                        np.random.shuffle(index2)
                        currentSet = currentSet[index2]
                        currentSetTask = currentSetTask[index2]

                        trainingCount = int(np.shape(currentSet)[0]/self.batch_size)
                        for j in range(trainingCount):

                            batch_z = np.random.normal(size=[self.batch_size, self.z_dim])

                            batchImage = currentSet[j * self.batch_size:(j + 1) * self.batch_size]
                            batchTask = currentSetTask[j * self.batch_size:(j + 1) * self.batch_size]

                            if j % 5 == 0:
                                # update G and Q network
                                _, g_loss = self.sess.run(
                                    [self.g_optim1, self.g_loss],
                                    feed_dict={self.inputs: batchImage, self.z: batch_z, self.d: batchTask})

                            # update D network
                            _, d_loss = self.sess.run([self.d_optim1, self.d_loss],
                                                      feed_dict={self.inputs: batchImage,
                                                                 self.z: batch_z, self.d: batchTask})

                            # Training student
                            _ = self.sess.run(
                                self.vae_optim,
                                feed_dict={self.inputs: batchImage, self.z: batch_z, self.gan_inputs: batchImage,self.d:batchTask})

                            _ = self.sess.run(
                                self.classifier2,
                                feed_dict={self.inputs: batchImage, self.z: batch_z, self.gan_inputs: batchImage,self.d:batchTask})

                            if j % 100 == 0:
                                print("save the image")
                                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
                                gan1 = self.sess.run(self.GenerationArr[np.shape(self.GenerationArr)[0] - 1],
                                                     feed_dict={self.z: batch_z})
                                ims_cv("results/" + "BBTest_" + str(m) + "_" + str(kk) + "_" + str(j) + ".png",
                                       merge2(gan1[:64], [8, 8]))

                        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                          .format(0, 0, 0, 0, g_loss, d_loss, self.componentCount, ))

            #Save model
            #Evaluation
            modelName = "MyFinalLVAEGAN_"
            dataName = "CelebAtoCACD_"

            myStr = modelName + dataName
            self.saver.save(sess, "model/" + myStr)

            for myi in range(10):
                sample = self.GenerateByCount(64)
                mySamples = sample

                name = modelName+dataName+ "Gen_2_" + str(myi) + ".png"
                name_small =  modelName+dataName+ "Gen_Teacjer_Small_2_" + str(myi) + ".png"
                ims_cv(os.path.join("results/", name), merge2(mySamples, [8, 8]))
                ims_cv(os.path.join("results/", name_small), merge2(mySamples[0:8], [2, 4]))


            # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

            testX1 = self.celebaTesting
            testX2 = cacdTesting

            # Create CelebA reconstructions
            test_data = testX1[0:64]

            batch = test_data[0:64]
            #reco = self.sess.run(self.Reco,fe)
            myReco = self.sess.run(self.Reco,feed_dict={self.inputs:batch})
            #myReco = Transfer_To_Numpy(reco)
            # myReco = merge2(myReco, [8, 8])

            realBatch = batch

            name = modelName+dataName+ "_RealCeleba_" + str(0) + ".png"
            name_small = modelName+dataName+"_RealCeleba_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(realBatch, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(realBatch[0:16], [2, 8]))

            name = modelName+dataName+"RecoCeleba_" + str(0) + ".png"
            name_small = modelName+dataName+"PRecoCeleba_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(myReco, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(myReco[0:16], [2, 8]))

            # Create Chair reconstruction
            test_data = testX2[0:64]

            batch = test_data[0:64]
            myReco = self.sess.run(self.Reco,feed_dict={self.inputs:batch})
            # myReco = merge2(myReco, [8, 8])

            realBatch = batch

            name = modelName+dataName+"_RealChair_" + str(0) + ".png"
            name_small = modelName+dataName+"_RealChair_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(realBatch, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(realBatch[0:8], [2, 4]))

            name = modelName+dataName+"_RecoChair_" + str(0) + ".png"
            name_small = modelName+dataName+"r_RecoChair_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(myReco, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(myReco[0:8], [2, 4]))

            # Create mixing reconstruction
            test_data = testX1[0:32]
            test_data = np.concatenate((test_data,testX2[0:32]),axis=0)

            n_examples_2 = np.shape(test_data)[0]
            index2 = [i for i in range(n_examples_2)]
            np.random.shuffle(index2)
            test_data = test_data[index2]

            batch = test_data[0:64]
            myReco = self.sess.run(self.Reco,feed_dict={self.inputs:batch})

            realBatch = batch

            name = modelName+dataName+ "_RealMix_" + str(0) + ".png"
            name_small = modelName+dataName+"_RealMix_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(realBatch, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(realBatch[0:8], [2, 4]))

            name = modelName+dataName+"_RecoMix_" + str(0) + ".png"
            name_small = modelName+dataName+"_RecoMix_Small_" + str(0) + ".png"

            ims_cv(os.path.join("results/", name), merge2(myReco, [8, 8]))
            ims_cv(os.path.join("results/", name_small), merge2(myReco[0:8], [2, 4]))

            #evaluation
            # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)
            recoData1 = self.Return_Reconstructions(testX1)
            recoData2 = self.Return_Reconstructions(testX2)
            testX1_ = testX1[0:np.shape(recoData1)[0]]
            testX2_ = testX2[0:np.shape(recoData2)[0]]

            # testing = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)
            # recoData = torch.tensor(recoData).cuda().to(device=device, dtype=torch.float)
            fid1 = self.FID_Evaluation(testX1_, recoData1)
            fid2 = self.FID_Evaluation(testX2_, recoData2)

            print(fid1)
            print(fid2)
            print((fid1 + fid2) / 2.0)

            print("Generation")
            x1 = testX1[0:2000]
            x2 = testX1[0:2000]
            xx = np.concatenate((x1,x2),axis=0)

            newGenerated = self.GenerateByCount(2000)
            xx = xx[0:np.shape(newGenerated)[0]]

            x1 = x1[0:np.shape(newGenerated)[0]]
            x2 = x2[0:np.shape(newGenerated)[0]]

            fid1 = self.FID_Evaluation(x1, newGenerated)
            fid2 = self.FID_Evaluation(x2, newGenerated)
            sumfid = fid1 + fid2
            sumfid = sumfid / 2.0
            print(fid1)
            print(fid2)
            print(sumfid)


infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
#infoMultiGAN.test()
