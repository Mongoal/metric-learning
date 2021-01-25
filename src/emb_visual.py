import  tensorflow as tf
from tensorboard.plugins import projector
import os
import h5py
import numpy as np
exp_name = '20201216-203244'
exp_name = '20201215-221824'
exp_name = '20210107-202542'
h5_filename = 'lte_pcl_20db_300_20201215-205653_v2.h5_embs.h5'
h5_filename = '20210107-202542_lte_pcl_20db_300_20201215-205653_v2.h5.h5'
meta_file = f'{h5_filename}_meta.tsv'
nselected = 100000
log_dir = './logs/visual/'+exp_name
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
path_h5 = os.path.join(exp_name,h5_filename)
path_metadata = os.path.join(log_dir, meta_file)

def visualisation(final_result):
    # 定义一个新向量保存输出层向量的取值
    y = tf.Variable(final_result, name="emb")
    # 定义日志文件writer
    summary_writer = tf.summary.FileWriter(log_dir)

    # ProjectorConfig帮助生成日志文件
    config = projector.ProjectorConfig()
    # 添加需要可视化的embedding
    embedding = config.embeddings.add()
    # 将需要可视化的变量与embedding绑定
    embedding.tensor_name = y.name

    # 指定embedding每个点对应的标签信息，
    # 这个是可选的，没有指定就没有标签信息
    embedding.metadata_path = meta_file
    # 指定embedding每个点对应的图像，
    # # 这个文件也是可选的，没有指定就显示一个圆点
    # embedding.sprite.image_path = sprite_file
    # # 指定sprite图中单张图片的大小
    # embedding.sprite.single_image_dim.extend([28, 28])

    # 将projector的内容写入日志文件
    projector.visualize_embeddings(summary_writer, config)

    # 初始化向量y，并将其保存到checkpoints文件中，以便于TensorBoard读取
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, 'model'), 1)
    summary_writer.close()


h5f = h5py.File(path_h5,'r')
embs = h5f['embs'].value
labels = h5f['labels'].value

selected = np.random.permutation(len(labels))[:nselected]


with open(path_metadata, 'w') as f:
    f.write('Index\tLabel\n')
    for index in selected:
        f.write('{}\t{}\n'.format(index, labels[index]))

visualisation(embs[selected])

