import numpy as np
import optimizer
import sinkhorn_ops
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage.transform import rescale, resize, downscale_local_mean


from scipy.optimize import linear_sum_assignment

from time import time

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data_train = input_data.read_data_sets('/tmp/', one_hot=True).train
data_test = input_data.read_data_sets('/tmp/', one_hot=True).test

from scipy.special import logsumexp


def sinkhorn(log_alpha, n_iters=20):
    for n in range(n_iters):
        log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
        log_alpha -= logsumexp(log_alpha, axis=0, keepdims=True)
        if n % 10000 == 1:
            print(n)
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)


def batch_split(batch, n_squares_side, n_channels=1):
    if (n_channels == 1):
        # side = int(np.sqrt(batch.shape[1]))
        side = batch.shape[1]
    else:
        side = batch.shape[1]
    batch_size = batch.shape[0]
    n_squares = n_squares_side ** 2

    batch = np.reshape(batch, [-1, side, side, n_channels])
    batch = np.reshape(batch, [batch_size, n_squares_side, int(side / n_squares_side), side, n_channels])
    batch = np.transpose(batch, [0, 2, 1, 3, 4])
    batch = np.reshape(batch, [batch_size, int(side / n_squares_side), n_squares, int(side / n_squares_side), n_channels])
    batch = np.transpose(batch, [0, 2, 1, 3, 4])
    return batch


def stack_batch_split(batch):
    return np.reshape(batch, [batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]])


def unflatten_batch(batch, n_channels=1):
    side_square = int(np.sqrt(batch.shape[2] / n_channels))
    return np.reshape(batch, [batch.shape[0], batch.shape[1], side_square, side_square, n_channels])


def join_batch_split(batch):
    batch_size = batch.shape[0]
    n_squares = batch.shape[1]
    side_quare = batch.shape[2]
    n_channels = batch.shape[4]
    n_squares_side = int(np.sqrt(n_squares))
    batch = np.transpose(batch, [0, 1, 3, 2, 4])
    batch = np.reshape(batch, [batch_size, n_squares_side, side_square * n_squares_side, side_square, n_channels])
    batch = np.transpose(batch, [0, 1, 3, 2, 4])
    batch = np.reshape(batch, [batch_size, 1, side_square * n_squares_side, side_square * n_squares_side, n_channels])
    batch = np.reshape(batch, [batch_size, side_square * n_squares_side, side_square * n_squares_side, n_channels])
    return batch


def resized_dims(n_squares_side):
    if (n_squares_side == 2):
        side = 28
        side_square = 14
    if (n_squares_side == 3):
        side = 27
        side_square = 9
    if (n_squares_side == 4):
        side = 28
        side_square = 7
    if (n_squares_side == 5):
        side = 30
        side_square = 6
    if (n_squares_side == 6):
        side = 30
        side_square = 5
    if (n_squares_side == 7):
        side = 28
        side_square = 4
    if (n_squares_side == 8):
        side = 32
        side_square = 4
    if (n_squares_side == 9):
        side = 27
        side_square = 3
    if (n_squares_side == 16):
        side = 80
        side_square = 5
    if (n_squares_side == 14):
        side = 196
        side_square = 14
    if (n_squares_side == 18):
        side = 8 * 18
        side_square = 8
    if (n_squares_side == 20):
        side = 7 * 20
        side_square = 7
    if (n_squares_side == 25):
        side = 25 * 6
        side_square = 6
    if (n_squares_side == 30):
        side = 30 * 5
        side_square = 5
    if (n_squares_side == 37):
        side = 37 * 4
        side_square = 4
    if (n_squares_side == 45):
        side = 45 * 3
        side_square = 3
    return side, side_square


def resize_batch_color(batch, side_new, n_channels):
    batch_new = np.zeros((batch.shape[0], side_new, side_new, n_channels))
    side = int(np.sqrt(batch.shape[1]))
    for i in range(batch.shape[0]):
        for c in range(n_channels):
            a = resize(batch[i, :, :, c], [side_new, side_new])

            a = a / 255.0
            batch_new[i, :, :, c] = a
    return batch_new


def soft_to_hard(soft_perm):
    a, b = linear_sum_assignment(-soft_perm)
    ma = np.zeros((np.shape(soft_perm)))
    for i in range(soft_perm.shape[0]):
        ma[i, b[i]] = 1
    return ma


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def create_z():
    # create the matrix of log_alpha, that will later will converted into a soft permutation
    # this relies on some NN processing (convolutional), see below
    fc = tf.contrib.layers.fully_connected
    flatten = tf.contrib.layers.flatten
    dropout = tf.contrib.layers.dropout

    def conv(input_image, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", bias_shape,
                                 initializer=tf.constant_initializer(0.0))
        convolutional = tf.nn.conv2d(input_image, weights,
                                     strides=[1, 1, 1, 1],
                                     padding="SAME")
        out_relu = tf.nn.relu(convolutional + biases)
        out_maxpool = tf.nn.max_pool(out_relu,
                                     ksize=[1, stride, stride, 1],
                                     strides=[1, stride, stride, 1],
                                     padding="SAME")
        return out_maxpool

    def conv_and_fc(input_image):
        with tf.variable_scope("conv1"):
            conv_output = conv(input_image, [rfield_size, rfield_size, n_channels, n_units], [n_units])
        fully_connected_output = dropout(
            tf.cast(fc(flatten(conv_output), n_dim_z * n_squares, activation_fn=None), tf.float32),
            keep_prob)
        return fully_connected_output

    with tf.variable_scope("model_params"):
        z0 = conv(real_entire_tiled, [rfield_size, rfield_size, n_channels, n_units], [n_units])
    #with tf.variable_scope('aa'):
        #z1 = flatten(z0)
        z1 = fc(flatten(z0), 10)
        #z1 = fc(flatten(z0), n_dim_z, activation_fn=None)
        #z = tf.reshape(z1, [-1, n_squares, n_dim_z])

    # with tf.variable_scope('bb'):
    #   z = tf.reshape(z1, [-1, n_squares, n_dim_z])
    #         sq = tf.reduce_sum(z **2, axis=2, keepdims=True)
    #         A = tf.tile(sq, [1, 1, n_squares])
    #         B = tf.tile(tf.transpose(sq, [0,2 ,1]), [1, n_squares, 1])
    #         C = -2*tf.matmul(z, tf.transpose(z, [0, 2, 1]))
    #         cost = A+B +C
    return z0, z1


#Define model params
batch_size = 5
n_iter_sinkhorn = 10

temp = 2.0

#mnist data
n_squares_side = 9
lr = 0.0001
n_channels = 1
rfield_size = 3
stride = 4
n_units = 1
keep_prob = 1.0
side_real = 28
opt = 'sgd'
samples_per_num = 1
n_squares = n_squares_side **2
n_gromov = 1
side, side_square = resized_dims(n_squares_side)
n_dim = int(side_square*side_square*n_channels)
print(n_dim)
n_dim_z = 3

noise_factor = 0
np.random.seed(1)

ims0,_=data_train.next_batch(1)
ims0 = np.expand_dims(np.reshape(ims0, [-1, side_real,side_real]),axis=3)
ims0[ims0>0.5]=1
ims0[ims0<0.5]=0



nx = np.nansum(ims0)
prop = nx/(side**2)*np.random.uniform(0.8,1.2)
pieces_split = np.zeros((batch_size, n_squares, side_square, side_square, 1))
for j in range(int(n_squares*prop)):
    pieces_split[:,j,:,:] = 1

scrambled_pieces_split = np.zeros(pieces_split.shape)


np.random.seed(1)
perm = np.random.permutation(n_squares)

scrambled_pieces_split[:,:, :, :] = pieces_split[:, perm, :, :]
stacked_scrambled_pieces_split = stack_batch_split(scrambled_pieces_split)

#Now we define the main TF variables


real_entire = tf.placeholder(tf.float32,[None, side_real, side_real, n_channels])
real_entire_tiled = tf.tile(real_entire,[samples_per_num,1,1,1])

temperature = tf.constant(temp, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
fc = tf.contrib.layers.fully_connected

# Now we define the main TF variables
from copy import deepcopy


z0, z = create_z()



ims,_=data_train.next_batch(batch_size)
ims = np.expand_dims(np.reshape(ims, [-1, 28,28]),axis=3)
ims[ims>0.5]=1
ims[ims<0.5]=0




np_x = resize_batch_color(ims, side, n_channels)

real_images_split = batch_split(np_x, n_squares_side, n_channels)
stacked_real_images_split = stack_batch_split(real_images_split)

# print(real_images_split.shape)
# print(scrambled_pieces_split.shape)
# print(stacked_scrambled_pieces_split.shape)
# print(stacked_real_images_split.shape)
init_op=tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init_op)
new = tf.trainable_variables()
print(new)
print(ims.shape)
for i in range(1):
    [zz0, zz1,rr] = sess.run([z0,z,real_entire_tiled],{real_entire:ims})
# for i in range(1):
#     [o,o2,loss, _] = sess.run([ordered_inf,ordered_inf2, l2s_diff, train_op],
#                                                                               {)
    print(zz0.shape)
    print(zz1.shape)

