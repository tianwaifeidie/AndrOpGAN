#coding:utf-8

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tensorflow.contrib import learn
import datetime

def main():
    stage = 'train'
    # stage = 'valid'
    #----load data----
    feat_ben = np.load('/home/amax/***/wgan2/ben_original_feature.npy')
    feat_mal = np.load('/home/amax/***/wgan2/mal_original_feature.npy')
    feat_ben = feat_ben[:500,:]
    feat_mal = feat_mal[:800,:]



    all_feat = np.concatenate((feat_ben, feat_mal), axis=0)


    print('all feat shape:{}'.format(all_feat.shape))

    label_mal = []
    for i in range(len(feat_mal)):
        label_mal.append([1])
    label_mal = np.array(label_mal)

    label_ben = []
    for i in range(len(feat_ben)):
        label_ben.append([0])
    label_ben = np.array(label_ben)

    all_label = np.concatenate((label_ben, label_mal), axis=0)
    all_data = np.concatenate((all_feat, all_label), axis=1)

    # print('all data shape:{}'.format(all_data.shape))

    np.random.shuffle(all_data)
    num_sample = len(all_data)

    #todo:如果想增加时使用的样本数，请修改此处的数值，* 后面的数代表了训练用样本的比例，数越小，用于测试的样本越多
    if stage == "train":
        thr = (int)(num_sample / 10 * 9)
    if stage == "valid":
        thr = (int)(num_sample / 10 * 1)

    train_data = all_data[0:thr, :]
    valid_data = all_data[thr:, :]

    train_feat = train_data[:, :-1]
    train_lab = train_data[:, -1]
    # print('train_feat shape:{}'.format(train_feat.shape))

    valid_feat = valid_data[:, :-1]
    valid_lab = valid_data[:, -1]

    train_one_hot_label = np.zeros([len(train_lab), 2])
    for idx in range(len(train_lab)):
        train_one_hot_label[idx][int(train_lab[idx])] = 1

    valid_one_hot_label = np.zeros([len(valid_lab), 2])
    for idx in range(len(valid_lab)):
        valid_one_hot_label[idx][int(valid_lab[idx])] = 1


    #----model construct----
    feat_size = len(train_feat[1])
    tf.reset_default_graph()
    network = input_data(shape=[None, 1, 45], name='input')
    branch1 = conv_1d(network, 256, 8, padding='same', activation='selu', regularizer="L2")
    branch2 = conv_1d(network, 256, 16, padding='same', activation='selu', regularizer="L2")
    branch3 = conv_1d(network, 256, 32, padding='same', activation='selu', regularizer="L2")
    # branch4 = conv_1d(network, 256, 64, padding='same', activation='selu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.9)
    # network = fully_connected(network, 1024, activation='softmax')
    # network = fully_connected(network, 512, activation='softmax')
    final_network = fully_connected(network, 2, activation='softmax')

    final_network = regression(final_network,
                               optimizer='adam',
                               learning_rate=0.0001,
                               loss='categorical_crossentropy',
                               name='target')

    model = tflearn.DNN(final_network)

    model_path = 'model/cnn2.tfl'

    if stage =='train':
    # if True:
    #----train model----
        try:
            print(train_feat.shape)
            # exit(0)
            train_data = train_feat.reshape([-1,1,45])
            model.fit(train_data, train_one_hot_label, n_epoch=20, validation_set=0.1,
                      show_metric=True, batch_size=2)
        except StopIteration as e:
            print('complete')

        model.save(model_path)

    if stage == 'valid':
    #----model test----
        model.load(model_path)
        total_rusult = np.zeros([1, 2])
        valid_sample_number = len(valid_feat)
        starttime = datetime.datetime.now()
        for index in range(valid_sample_number):
            result = model.predict(valid_feat[index:index + 1, :].reshape([1,1,45]))
            total_rusult = np.concatenate((total_rusult, result))
            # print(index)
        endtime = datetime.datetime.now()
        time = (endtime - starttime)
        total_rusult = total_rusult[1:, :]

        safe_hat = np.argmax(total_rusult, axis=1)
        safe_true = np.int64(valid_lab)
        safe_tp = np.sum(safe_hat * safe_true)
        safe_tn = np.sum((1 - safe_hat) * (1 - safe_true))
        safe_fp = np.sum((safe_hat) * (1 - safe_true))
        safe_fn = np.sum((1 - safe_hat) * (safe_true))

        precesion = safe_tp /(safe_tp + safe_fp)
        recall = safe_tp / (safe_tp + safe_fn)

        print('-----------------')
        print(safe_hat)
        print('-----------------')
        print(safe_true)
        print('-----------------')

        print('safe_tp:{}'.format(safe_tp))
        print('safe_tn:{}'.format(safe_tn))
        print('safe_fp:{}'.format(safe_fp))
        print('safe_fn:{}'.format(safe_fn))
        print('precesion:{}'.format(precesion))
        print('recall:{}'.format(recall))
        print('time:{}'.format(time))


if __name__ == '__main__':
    main()
