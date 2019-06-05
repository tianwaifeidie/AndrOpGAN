# -*- coding: utf-8 -*-
"""
based on DCGAN example of tflearn
for paper3
###############################
##To generate feature library##
###############################
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import tflearn



class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self,val_loss_thresh):
        self.val_loss_thresh = val_loss_thresh
    def on_epoch_end(self,training_state):
        print('---judging loss---')
        if training_state.loss_value <= self.val_loss_thresh:
            raise StopIteration
    def on_train_end(self,training_state):
        print('model saved')


def private_loss(y_pred, y_true):

    abs = tf.abs(y_pred)
    div = tf.div(y_pred,abs,name=None)
    middle = tf.subtract(div,tf.abs(div),name=None)
    loss = tf.abs(tf.reduce_sum(middle))/100
    return loss

# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = tflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 1, 5, activation='sigmoid')
        x = tflearn.fully_connected(x, n_units=45)
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 9, 5, 1])
        x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.conv_2d(x, 128, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.fully_connected(x, 1024, activation='tanh')
        x = tflearn.fully_connected(x, 2)
        x = tf.nn.softmax(x)
        return x


def main():
    stage ='train'
    # stage = 'gen'
    model_path = 'gan_model/gan_paper_3_private_loss.tfl'


    if stage == 'train':
        X = np.load('ben_original_feature.npy')
        X_1 = X[500:,:45]
        X = X[500:,:]
        X = X[:,:45]
        X = np.reshape(X, newshape=[-1, 9, 5, 1])
        z_dim = 20 # Noise data points
        total_samples = len(X)

        # Input Data
        gen_input = tflearn.input_data(shape=[None, z_dim], name='input_gen_noise')
        input_disc_noise = tflearn.input_data(shape=[None, z_dim], name='input_disc_noise')
        input_disc_real = tflearn.input_data(shape=[None, 9, 5, 1], name='input_disc_real')


        #Output Data of Generator



        # Build Discriminator
        disc_fake = discriminator(generator(input_disc_noise))
        disc_real = discriminator(input_disc_real, reuse=True)
        disc_net = tf.concat([disc_fake, disc_real], axis=0)

        # Build Stacked Generator/Discriminator
        gen_net = generator(gen_input, reuse=True)
        stacked_gan_net = discriminator(gen_net, reuse=True)

        # Build Training Ops for both Generator and Discriminator.
        # Each network optimization should only update its own variable, thus we need
        # to retrieve each network variables (with get_layer_variables_by_scope).
        disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')

        # We need 2 target placeholders, for both the real and fake image target.
        disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'], shape=[None, 2])
        gen_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'], shape=[None, 2])
        disc_model = tflearn.regression(disc_net, optimizer='adam',
                                        placeholder=disc_target,
                                        loss='categorical_crossentropy',
                                        trainable_vars=disc_vars,
                                        batch_size=64, name='target_disc',
                                        op_name='DISC')

        gen_vars = tflearn.get_layer_variables_by_scope('Generator')
        gan_model = tflearn.regression(stacked_gan_net, optimizer='adam',
                                       loss='categorical_crossentropy',
                                       trainable_vars=gen_vars,
                                       batch_size=64, name='target_gen',
                                       op_name='GEN')


        gen_to_train = tflearn.regression(gen_net, optimizer='adam',
                                       loss= private_loss,
                                       trainable_vars=gen_vars,
                                       batch_size=64, name='gen_train',
                                       op_name='GEN')

        # Define GAN model
        gan = tflearn.DNN(gan_model)


        # Training
        # Prepare input data to feed to the discriminator
        disc_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
        # Prepare target data to feed to the discriminator (0: fake image, 1: real image)
        y_disc_fake = np.zeros(shape=[total_samples])
        y_disc_real = np.ones(shape=[total_samples])
        y_disc_fake = tflearn.data_utils.to_categorical(y_disc_fake, 2)
        y_disc_real = tflearn.data_utils.to_categorical(y_disc_real, 2)

        # Prepare input data to feed to the stacked generator/discriminator
        gen_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
        # Prepare target data to feed to the discriminator
        # Generator tries to fool the discriminator, thus target is 1 (e.g. real images)
        y_gen = np.ones(shape=[total_samples])
        y_gen = tflearn.data_utils.to_categorical(y_gen, 2)



        gan.fit(X_inputs={'input_gen_noise': gen_noise,
                          'input_disc_noise': disc_noise,
                          'input_disc_real': X,
                          },
                Y_targets={'target_gen': y_gen,
                           'target_disc_fake': y_disc_fake,
                           'target_disc_real': y_disc_real,
                           'gen_train': X_1 },
                n_epoch=10)

        print('--GAN training finish--')



        # gen_dnn = tflearn.DNN(gen_net,session=gan.session)
        # gen_dnn.fit(X_inputs={'gen_input':gen_noise},Y_targets={'gen_train':X},n_epoch=10)


        gan.save(model_path)


    if stage == 'gen':
        gen_amount = 13000
        z_dim = 20  # Noise data points

        # Input Data
        gen_input = tflearn.input_data(shape=[None, z_dim], name='input_gen_noise')
        input_disc_noise = tflearn.input_data(shape=[None, z_dim], name='input_disc_noise')
        input_disc_real = tflearn.input_data(shape=[None, 9, 5, 1], name='input_disc_real')

        # Build Discriminator
        disc_fake = discriminator(generator(input_disc_noise))
        disc_real = discriminator(input_disc_real, reuse=True)
        disc_net = tf.concat([disc_fake, disc_real], axis=0)
        # Build Stacked Generator/Discriminator
        gen_net = generator(gen_input, reuse=True)
        stacked_gan_net = discriminator(gen_net, reuse=True)

        # Build Training Ops for both Generator and Discriminator.
        # Each network optimization should only update its own variable, thus we need
        # to retrieve each network variables (with get_layer_variables_by_scope).
        disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')

        # We need 2 target placeholders, for both the real and fake image target.
        disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                                shape=[None, 2])
        disc_model = tflearn.regression(disc_net, optimizer='adam',
                                        placeholder=disc_target,
                                        loss='categorical_crossentropy',
                                        trainable_vars=disc_vars,
                                        batch_size=64, name='target_disc',
                                        op_name='DISC')

        gen_vars = tflearn.get_layer_variables_by_scope('Generator')
        gan_model = tflearn.regression(stacked_gan_net, optimizer='adam',
                                       loss='categorical_crossentropy',
                                       trainable_vars=gen_vars,
                                       batch_size=64, name='target_gen',
                                       op_name='GEN')

        # Define GAN model, that output the generated images.
        gan = tflearn.DNN(gan_model)
        gan.load(model_path)

        # Create another model from the generator graph to generate some samples
        # for testing (re-using same session to re-use the weights learnt).
        gen = tflearn.DNN(gen_net, session=gan.session)
        g = np.zeros((1,45))
        for i in range(int(gen_amount/10)):
            z = np.random.uniform(-1., 1., size=[10, z_dim])
            g_part = np.array(gen.predict({'input_gen_noise': z}))
            g = np.concatenate((g,g_part),axis=0)

        # np.save('gen_result.npy', g[1:,:])
        # data = np.load('gen_result.npy')

        data = g[1:,:]
        sample_num = len(data)
        print('sample_number:{}'.format(sample_num))
        result2 = []
        data = np.abs(data)
        for i in range(len(data)):
            # if np.sum(data[i]) <= 1 and np.sum(data[i]) >= 0.9:
            if np.sum(data[i]) <= 100000:
                result2.append(data[i])
        result2 = np.array(result2)
        print('get:{}'.format(len(result2)))
        for i in range(sample_num):
            result2[i, :-1] = result2[i, :-1] / np.sum(result2[i, :-1])

        np.save('result2_'+str(gen_amount)+'.npy', result2)


if __name__ == '__main__':
    main()