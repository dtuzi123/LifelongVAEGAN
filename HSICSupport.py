# Tensorflow implementation of HSIC
# Refers to original implementations
# https://github.com/kacperChwialkowski/HSIC
# https://cran.r-project.org/web/packages/dHSIC/index.html

from scipy.special import gamma
import tensorflow as tf
import numpy as np

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel,
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d + 1)) / gamma(0.5 * d)
    return 1. / (2. * gz ** 2)


def K(x1, x2, gamma=1.):
    dist_table = tf.expand_dims(x1, 0) - tf.expand_dims(x2, 1)
    return tf.transpose(tf.exp(-gamma * tf.reduce_sum(dist_table ** 2, axis=2)))

def hsic_individual(z, s):
    # use a gaussian RBF for every variable

    d_z = z.get_shape().as_list()[1]
    d_s = s.get_shape().as_list()[1]

    zz = K(z, z, gamma=bandwidth(d_z))
    ss = K(s, s, gamma=bandwidth(d_s))

    hsic = 0
    hsic += tf.reduce_mean(zz * ss)
    hsic += tf.reduce_mean(zz) * tf.reduce_mean(ss)
    hsic -= 2 * tf.reduce_mean(tf.reduce_mean(zz, axis=1) * tf.reduce_mean(ss, axis=1))
    return tf.sqrt(hsic)

def dHSIC(list_variables):
    for i, z_j in enumerate(list_variables):
        k_j = K(z_j, z_j, gamma=bandwidth(z_j.get_shape().as_list()[1]))
        if i == 0:
            term1 = k_j
            term2 = tf.reduce_mean(k_j)
            term3 = tf.reduce_mean(k_j, axis=0)
        else:
            term1 = term1 * k_j
            term2 = term2 * tf.reduce_mean(k_j)
            term3 = term3 * tf.reduce_mean(k_j, axis=0)
    dhsic = tf.sqrt(tf.reduce_mean(term1) + term2 - 2 * tf.reduce_mean(term3))
    return dhsic

def Give_Reconstruction(x1,x2):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x1 - x2)))
    return reconstruction_loss

def Give_Reconstruction_3(x1,x2):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x1 - x2), [1, 2, 3]))

def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits))

def differentiable_sample_1(logits, cats_range, temperature=.1):
    noise = np.random.gumbel(size=len(logits))
    logits_with_noise = softmax((logits+noise)/temperature)
    #print(logits_with_noise)
    sample = np.sum(logits_with_noise*cats_range)
    return sample

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def my_gumbel_softmax_sample(logits,cats_range, temperature=0.1):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  logits_with_noise = tf.nn.softmax( y / temperature)
  return logits_with_noise

def gumbel_softmax_sample(logits,cats_range, temperature=0.1):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  logits_with_noise = tf.nn.softmax( y / temperature)
  samples = tf.reduce_sum(logits_with_noise * cats_range,1)
  return samples

def gumbel_softmax_sample2(logits,cats_range,shape,temperature=0.1):
    y = logits + sample_gumbel(shape)
    logits_with_noise = tf.nn.softmax(y / temperature)
    samples = tf.reduce_sum(logits_with_noise * cats_range,1)
    return samples

def gumbel_softmax_sample3(logits1,logist2,cats_range,shape,temperature=0.1):
    y1 = logits1 + sample_gumbel(shape)
    y2 = logist2 + sample_gumbel(shape)
    y = tf.stack((y1,y2),1)
    logits_with_noise = tf.nn.softmax(y / temperature)

    samples = tf.reduce_sum(logits_with_noise * cats_range,1)
    return samples

def gumbel_softmax_sample_3(logits1,logist2,cats_range,shape,temperature=0.1):
    y1 = logits1 + sample_gumbel(shape)
    y2 = logist2 + sample_gumbel(shape)
    #y = tf.stack((y1,y2),1)
    y = tf.concat((y1,y2),axis=1)

    logits_with_noise = tf.nn.softmax(y / temperature)

    samples = tf.reduce_sum(logits_with_noise * cats_range,1)
    return samples

def ConvertToString(arr,count):
    arr1 = []
    for i in range(count):
        t1 = arr[i]
        arr1.append(t1+'\n')
    return arr1

# dHSIC
# list_variables has to be a list of tensorflow tensors
# for i, z_j in enumerate(list_variables):
#     k_j = K(z_j, z_j, gamma=bandwidth(z_j.get_shape().as_list()[1]))
#     if i == 0:
#         term1 = k_j
#         term2 = tf.reduce_mean(k_j)
#         term3 = tf.reduce_mean(k_j, axis=0)
#     else:
#         term1 = term1 * k_j
#         term2 = term2 * tf.reduce_mean(k_j)
#         term3 = term3 * tf.reduce_mean(k_j, axis=0)
# dhsic = tf.sqrt(tf.reduce_mean(term1) + term2 - 2 * tf.reduce_mean(term3))