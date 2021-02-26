# 用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
# 预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# import detect_face
import math

import facenet

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
from prepare_v03 import myfft2, append_data_to_h5
import argparse

def hist_pd(data, n):
    hist, bin_edges = np.histogram(data, n)
    hist = hist / np.sum(hist) / (bin_edges[-1] - bin_edges[0]) * n
    return bin_edges[1:], hist

def match(vector,centers,threshold):
    diss = np.sqrt(np.sum(np.square(vector - centers), 1))
    matched = np.argmin(diss)
    if diss[matched] > threshold:
        matched = -1
    return matched


parser = argparse.ArgumentParser()

parser.add_argument('--modelname', type=str, default='20210107-202542')
parser.add_argument('--h5path', type=str, default='/userhome/database/lte_pcl_20db_300_20201215-205653_v2.h5')

args = parser.parse_args()

train_labels = [0, 2, 3, 6, 8, 9, 11, 12, 14, 15]
unknown_labels = list(set(range(27)) - set(train_labels))
modelname = '20201216-203244' #eval_triplet_h5_fig670644
modelname = args.modelname
h5path = args.h5path
batch_size = 128
modelpath = '/userhome/code/result/models/'+modelname
modelpath = ''+modelname
np.random.seed(seed=666)
h5f = h5py.File(h5path, 'r')
data = h5f['signal'][:]
labels = h5f['labels'][:]
dataset = facenet.get_dataset_idx_from_h5f(h5f, 'labels', 'imsi', include_labels=train_labels)
train_set, eval_set = facenet.split_dataset(dataset, 0.3, 1, 'SPLIT_IMAGES')
unknown_set = facenet.get_dataset_idx_from_h5f(h5f, 'labels', 'imsi', include_labels=unknown_labels)
nTrain = sum([len(specClass) for specClass in train_set])
nEval = sum([len(specClass) for specClass in eval_set])
nUnknown = sum([len(specClass) for specClass in unknown_set])
# test_set, test_indices = radar_io.get_h5dataset(abspath, 7, 'TEST')
# unknown_set, unknown_indices = radar_io.get_h5dataset(abspath, 7, 'UNKONWN')
num_train_class = len(train_set)
num_unknown_class = len(unknown_set)
# plt.figure()
# plt.imshow(images[1,:])
# plt.show()
# print('askhnauisd')
if not os.path.isdir(f'{modelname}/fig/'):
    os.makedirs(f'{modelname}/fig/',exist_ok=True)


if not os.path.exists(f'softmax_label_{os.path.basename(modelname)}_{os.path.basename(h5path)}.h5'):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = modelpath
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            input = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            logits =  tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
            softmax = tf.nn.softmax(logits)
            predict = tf.argmax(softmax)
            prob = tf.reduce_max(softmax,axis=1)
            print(logits.shape)
            nclass = softmax.shape[1]
            sfmx_array = np.zeros((len(data), nclass))
            for i in range(math.ceil(len(data) / batch_size)):
                indices = range(batch_size * i, min(batch_size * i + batch_size, len(data)))
                signals = data[indices]
                stfts = np.asarray([myfft2(s, 128, 255, 64, False) for s in signals])
                feed_dict = {input: stfts, phase_train_placeholder: False}
                sfmx_array[indices, :] = sess.run(softmax, feed_dict=feed_dict)

            h5emb = h5py.File(f'softmax_label_{os.path.basename(model)}_{os.path.basename(h5path)}.h5', 'w')
            append_data_to_h5(h5emb, sfmx_array, 'softmax')
            append_data_to_h5(h5emb, h5f['labels'].value, 'labels')
            h5emb.close()
            # # train_set : Run forward pass to calculate embeddings
            # emb_train =np.zeros((len(train_set), 128))
            # for i in range(math.ceil(len(train_set)/batch_size)):
            #     indices = range(batch_size*i, min(batch_size*i+batch_size,len(train_set)))
            #
            #     signals = data[image_paths]
            #     stfts = np.asarray([myfft2(s, 128, 255, 64, False) for s in signals])
            #
            #     feed_dict = {input: stfts, phase_train_placeholder: False}
            #     emb_train[indices,:] = sess.run(embeddings, feed_dict=feed_dict)
            #
            # # test_set : Run forward pass to calculate embeddings
            # emb_test =np.zeros((len(test_set), 128))
            # for i in range(math.ceil(len(test_set)/batch_size)):
            #     indices = range(batch_size*i, min(batch_size*i+batch_size,len(test_set)))
            #     feed_dict = {input: test_set[indices], phase_train_placeholder: False}
            #     emb_test[indices,:] = sess.run(embeddings, feed_dict=feed_dict)
            #
            # # train_set : Run forward pass to calculate embeddings
            # emb_unknown =np.zeros((len(unknown_set), 128))
            # for i in range(math.ceil(len(unknown_set)/batch_size)):
            #     indices = range(batch_size*i, min(batch_size*i+batch_size,len(unknown_set)))
            #     feed_dict = {input: unknown_set[indices], phase_train_placeholder: False}
            #     emb_unknown[indices,:] = sess.run(embeddings, feed_dict=feed_dict)
else:
    h5emb = h5py.File(f'{os.path.basename(modelpath)}_{os.path.basename(h5path)}.h5', 'r')
    sfmx_array = h5emb['softmax'][:]

'''
 evalation
'''
# train test unknown
probs = np.max(sfmx_array, axis=1)
pred = np.argmax(sfmx_array, axis=1)
p_thres = np.arange(0.5,1,0.01)
train_acc = np.zeros(len(p_thres))
test_acc = np.zeros(len(p_thres))
unknown_acc = np.zeros(len(p_thres))
for pi, p_thre in enumerate(p_thres):
    print(f'***************\np_thre = {p_thre}\n**************')
    is_unknown = probs < p_thre
    pred = np.where(is_unknown, -1, pred)
    train_pred_right = []
    test_pred_right = []
    unknown_pred_right = []
    print('\nTrain set')
    for i, x in enumerate(train_set):
        pred_right = labels[x.idxes] == pred[x.idxes]
        train_pred_right.append(pred_right)
        print(f'class {x.name} acc {np.mean(pred_right):<.3f}')

    print('\nTest set')
    for i, x in enumerate(eval_set):
        pred_right = labels[x.idxes] == pred[x.idxes]
        test_pred_right.append(pred_right)
        print(f'class {x.name} acc {np.mean(pred_right):<.3f}')

    print('\nUnknown set')
    for i, x in enumerate(unknown_set):
        pred_right = -1 == pred[x.idxes]
        unknown_pred_right.append(pred_right)
        print(f'class {x.name} acc {np.mean(pred_right):<.3f}')

    train_acc[pi] = np.mean(np.concatenate(train_pred_right))
    test_acc[pi] = np.mean(np.concatenate(test_pred_right))
    unknown_acc[pi] = np.mean(np.concatenate(unknown_pred_right))

    plt.plot(train_acc,label='train_acc',linestyle='--',color='r',marker='D')
    plt.plot(test_acc,label='test_acc',linestyle='-',color='b',marker='o')
    plt.plot(unknown_acclabel='unknown_acc',linestyle='-',color='g',marker='^')
    plt.xlabel('Softmax probability threshold')
    plt.ylabel('Accuracy')
    plt.imsave(f'{modelname}/fig/curve.png')

