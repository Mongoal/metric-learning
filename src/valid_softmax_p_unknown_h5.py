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
data = h5f['signal']
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


if not os.path.exists(f'{os.path.basename(modelpath)}_{os.path.basename(h5path)}.h5'):
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
            embs = np.zeros((len(data), 128))
            for i in range(math.ceil(len(data) / batch_size)):
                indices = range(batch_size * i, min(batch_size * i + batch_size, len(data)))
                signals = data[indices]
                stfts = np.asarray([myfft2(s, 128, 255, 64, False) for s in signals])
                feed_dict = {input: stfts, phase_train_placeholder: False}
                embs[indices, :] = sess.run(embeddings, feed_dict=feed_dict)

            h5emb = h5py.File(f'{os.path.basename(model)}_{os.path.basename(h5path)}.h5', 'w')
            append_data_to_h5(h5emb, embs, 'embs')
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
    embs = h5emb['embs'][:]

'''
 emb center
'''
# train test unknown
emb_center_train = np.zeros((num_train_class, 128))
emb_center_test = np.zeros((num_train_class, 128))
emb_center_unknown = np.zeros((num_unknown_class, 128))
for i, x in enumerate(train_set):
    emb_center_train[i, :] = np.mean(embs[x.idxes], 0)
    emb_center_train[i, :] /= np.linalg.norm(emb_center_train[i, :])
for i, x in enumerate(eval_set):
    emb_center_test[i, :] = np.mean(embs[x.idxes], 0)
    emb_center_test[i, :] /= np.linalg.norm(emb_center_test[i, :])
for i, x in enumerate(unknown_set):
    emb_center_unknown[i, :] = np.mean(embs[x.idxes], 0)
    emb_center_unknown[i, :] /= np.linalg.norm(emb_center_unknown[i, :])

'''
 distance among emb centers
'''
# among train set emb centers
dist_train_emb_center = np.zeros((num_train_class, num_train_class))
for i, m in enumerate(emb_center_train):
    for j, n in enumerate(emb_center_train):
        dist_train_emb_center[i, j] = np.sqrt(np.sum(np.square(m - n)))
# among unknown set emb centers
dist_unknown_emb_center = np.zeros((num_unknown_class, num_unknown_class))
for i, m in enumerate(emb_center_unknown):
    for j, n in enumerate(emb_center_unknown):
        dist_unknown_emb_center[i, j] = np.sqrt(np.sum(np.square(m - n)))

# between test set emb centers & train set emb centers
dist_test_train_emb_center = np.zeros((num_train_class, num_train_class))
for i, m in enumerate(emb_center_test):
    for j, n in enumerate(emb_center_train):
        dist_test_train_emb_center[i, j] = np.sqrt(np.sum(np.square(m - n)))

# between unknown set emb centers & train set emb centers
dist_unknown_train_emb_center = np.zeros((num_unknown_class, num_train_class))
for i, m in enumerate(emb_center_unknown):
    for j, n in enumerate(emb_center_train):
        dist_unknown_train_emb_center[i, j] = np.sqrt(np.sum(np.square(m - n)))

'''
bias to centers 
'''
dis_to_center = np.zeros(len(data))
# train
mean_bias_train = np.zeros(num_train_class)
std_bias_train = np.zeros(num_train_class)
for i, x in enumerate(train_set):
    dis_to_center[x.idxes] = np.sqrt(np.sum(np.square(embs[x.idxes] - emb_center_train[i, :]), 1))
    mean_bias_train[i] = np.mean(dis_to_center[x.idxes])
    std_bias_train[i] = np.std(dis_to_center[x.idxes])
    plt.clf()
    dis, a = hist_pd(dis_to_center[x.idxes], 50)
    plt.plot(dis, a)
    plt.vlines(dist_train_emb_center[i], 0, a.max())
    plt.title('bias to center, train ' + x.name + '\n' + 'mean = ' + str(mean_bias_train[i])[:5] + ', std = ' + str(
        std_bias_train[i])[:5])
    plt.savefig(f'{modelname}/fig/' + 'bias to center train ' + x.name + '.png')

# test
mean_bias_test = np.zeros(num_train_class)
std_bias_test = np.zeros(num_train_class)
for i, x in enumerate(eval_set):
    dis_to_center[x.idxes] = np.sqrt(np.sum(np.square(embs[x.idxes] - emb_center_test[i, :]), 1))
    mean_bias_test[i] = np.mean(dis_to_center[x.idxes])
    std_bias_test[i] = np.std(dis_to_center[x.idxes])
    plt.clf()
    dis, a = hist_pd(dis_to_center[x.idxes], 50)
    plt.plot(dis, a)
    plt.vlines(dist_train_emb_center[i], 0, a.max())
    plt.title('bias to center, test ' + x.name + '\n' + 'mean = ' + str(mean_bias_test[i])[:5] + ', std = ' + str(
        std_bias_test[i])[:5])
    plt.savefig(f'{modelname}/fig/' + 'bias to center test ' + x.name + '.png')

# unknown
mean_bias_unknown = np.zeros(num_unknown_class)
std_bias_unknown = np.zeros(num_unknown_class)
for i, x in enumerate(unknown_set):
    dis_to_center[x.idxes] = np.sqrt(np.sum(np.square(embs[x.idxes] - emb_center_unknown[i, :]), 1))
    mean_bias_unknown[i] = np.mean(dis_to_center[x.idxes])
    std_bias_unknown[i] = np.std(dis_to_center[x.idxes])
    plt.clf()
    dis, a = hist_pd(dis_to_center[x.idxes], 50)
    plt.plot(dis, a)
    plt.vlines(dist_unknown_emb_center[i], 0, a.max())
    plt.title('bias to center, unknown ' + x.name + '\n' + 'mean = ' + str(mean_bias_unknown[i])[:5] + ', std = ' + str(
        std_bias_unknown[i])[:5])
    plt.savefig(f'{modelname}/fig/' + 'bias to center unknown ' + x.name + '.png')

print('distance from unknown center to known center:\n', dist_unknown_train_emb_center)

for i, x in enumerate(unknown_set):
    tx = train_set[np.argmin(dist_unknown_train_emb_center[i])]
    dis_to_center[x.idxes] = np.sqrt(
        np.sum(np.square(embs[x.idxes] - emb_center_train[np.argmin(dist_unknown_train_emb_center[i]), :]), 1))
    mean_bias_unknown[i] = np.mean(dis_to_center[x.idxes])
    std_bias_unknown[i] = np.std(dis_to_center[x.idxes])
    plt.clf()
    dis, a = hist_pd(dis_to_center[x.idxes], 50)
    plt.plot(dis, a)
    dis, a = hist_pd(dis_to_center[tx.idxes], 50)
    plt.plot(dis, a)
    plt.title('distance to train center, unknown ' + x.name + ' & train ' + tx.name)
    plt.savefig(f'{modelname}/fig/' + 'distance to train center, unknown ' + x.name + ' & train ' + tx.name + '.png')

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    print(f'************************************************')
    print(f'************** threshold = {threshold} ***************')
    print(f'************************************************')
    print(f':::: train set ::::')
    true_positive = 0
    total_num = 0
    for i, x in enumerate(train_set):
        total_num += len(x)
        true_positive_cur_class = 0
        for idx in x.idxes:
            matched = match(embs[idx],emb_center_train,threshold)
            true_positive_cur_class += matched == i
        true_positive += true_positive_cur_class
        print(f'class {i} - IMSI {x.name} train set acc: {true_positive_cur_class/len(x):.3f}')
    print(f'total train set acc:{true_positive/total_num:.3f}')


    print(f':::: test set ::::')
    true_positive = 0
    total_num = 0
    for i, x in enumerate(eval_set):
        total_num += len(x)
        true_positive_cur_class = 0
        for idx in x.idxes:
            matched = match(embs[idx],emb_center_train,threshold)
            true_positive_cur_class += matched == i
        true_positive += true_positive_cur_class
        print(f'class {i} test set acc: {true_positive_cur_class/len(x):.3f}')
    print(f'total test set acc:{true_positive/total_num:.3f}')


    print(f':::: unknown set ::::')
    true_positive = 0
    total_num = 0
    for i, x in enumerate(unknown_set):
        total_num += len(x)
        true_positive_cur_class = 0
        for idx in x.idxes:
            matched = match(embs[idx],emb_center_train,threshold)
            true_positive_cur_class += matched == -1
        true_positive += true_positive_cur_class
        print(f'class {i} unknown set acc: {true_positive_cur_class/len(x):.3f}')
    print(f'total unknown set acc:{true_positive/total_num:.3f}')



# for x in train_indices:
#     for i, h in enumerate(np.random.permutation(len(x))[:3]):
#         plt.figure(figsize = (8,4))
#         plt.imshow(emb_train[x.idxes[h]].reshape(8, 16), cmap=plt.cm.Oranges)
#         plt.colorbar()
#         plt.title('train '+x.name+' '+str(i))
#         plt.savefig(f'{modelname}/fig/'+'train '+x.name+' '+str(i)+'.png')
#
# for x in unknown_indices:
#     for i, h in enumerate(np.random.permutation(len(x))[:3]):
#         plt.figure(figsize=(8, 4))
#         plt.imshow(emb_unknown[x.idxes[h]].reshape(8, 16), cmap=plt.cm.Oranges)
#         plt.colorbar()
#         plt.title('unknown ' + x.name + ' ' + str(i))
#         plt.savefig(f'{modelname}/fig/' + 'unknown ' + x.name + ' ' + str(i) + '.png')
# #
# emb_center_train = np.mean(emb_train,1)
# emb_center_test = np.mean(emb_train,1)
#
# # train_set, mean std
# num_class = num_train_class
# train_mean,train_std = np.zeros((num_class,num_class)),np.zeros((num_class,num_class))
# for m in range(num_class):
#     print('train_set %d is %s ' % (m ,train_indices[m].name))
#     indices = train_indices[m].indices
#     dist = []
#     for i in range( min(1000,len(indices))):
#         for j in range(i + 1, min(1000,len(indices))):
#             dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_train[indices[i], :], emb_train[indices[j], :])))))
#     train_mean[m,m] = np.mean(dist)
#     train_std[m,m] = np.std(dist)
# for m in range(num_class):
#     for n in range(num_class):
#         if m != n :
#             dist = []
#             for i in train_indices[m].indices[:min(1000,len(train_indices[m]))]:
#                 for j in train_indices[n].indices[:min(1000,len(train_indices[n]))]:
#                     dist.append(np.sqrt(
#                         np.sum(np.square(np.subtract(emb_train[i, :], emb_train[j, :])))))
#             train_mean[m, n] = np.mean(dist)
#             train_std[m, n] = np.std(dist)
#
#
# # train_test_dist, mean std
# num_class = num_train_class
# train_test_dist_mean,train_test_dist_std = np.zeros((num_class,num_class)),np.zeros((num_class,num_class))
#
# for m in range(num_class):
#     for n in range(num_class):
#         dist = []
#         for i in train_indices[m].indices[:min(1000,len(train_indices[m]))]:
#             for j in test_indices[n].indices[:min(1000,len(test_indices[n]))]:
#                 dist.append(np.sqrt(
#                     np.sum(np.square(np.subtract(emb_train[i, :], emb_test[j, :])))))
#         train_test_dist_mean[m, n] = np.mean(dist)
#         train_test_dist_std[m, n] = np.std(dist)
#
# # unknown_train_dist, mean std
#
# unknown_num_class = len(unknown_indices)
# unknown_train_dist_mean,unknown_train_dist_std = np.zeros((unknown_num_class,num_train_class)),np.zeros((unknown_num_class,num_train_class))
#
# for m in range(unknown_num_class):
#     for n in range(num_train_class):
#         dist = []
#         for i in unknown_indices[m].indices[:min(1000,len(unknown_indices[m]))]:
#             for j in train_indices[n].indices[:min(1000,len(train_indices[n]))]:
#                 dist.append(np.sqrt(
#                     np.sum(np.square(np.subtract(emb_unknown[i, :], emb_train[j, :])))))
#         unknown_train_dist_mean[m, n] = np.mean(dist)
#         unknown_train_dist_std[m, n] = np.std(dist)
#
#
#
#
#

# nrof_images = len(args.csv_paths)
#
# print('Images:')
# for i in range(nrof_images):
#     print('%1d: %s' % (i, args.csv_paths[i]))
# print('')
#
# # Print distance matrix
# print('Distance matrix')
# print('    ', end='')
# for i in range(nrof_images):
#     print('    %1d     ' % i, end='')
# print('')
# for i in range(nrof_images):
#     print('%1d  ' % i, end='')
#     for j in range(nrof_images):
#         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
#         print('  %1.4f  ' % dist, end='')
#     print('')
