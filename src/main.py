from tqdm import tqdm

from h5data_reader import H5DataReader
from prepare_v03 import myfft2
from multi_preprocess import MultiProcessDataHandler
import numpy as np
import matplotlib.pyplot as plt
import time
def preprocess(data):
    # print(data)
    x = data[0]
    y = data[1]

    return [ myfft2(xx,128, 255, 64, False) for xx in x] , y

if __name__ == '__main__':
    s = time.time()
    print(time.time()-s)
    h5path = r'F:\python\version_20.04.07_git\lte_pcl_20db_100_df_20201208_v2.h5'
    reader = H5DataReader(h5path,data_key='signal')
    mpdh = MultiProcessDataHandler(preprocess,processes=7)
    for _ in range(100):
        data = reader.get_shuffle_data(64)
        mpdh.handle(data)
    for _ in tqdm(range(100)):
        mpdh.get_result()
    print(time.time()-s)
    s = time.time()

    h5path = r'F:\python\version_20.04.07_git\lte_pcl_20db_100_df_20201208_v2.h5'
    reader = H5DataReader(h5path,data_key='signal')
    for _ in tqdm(range(100)):
        preprocess(reader.get_shuffle_data(64))

    print(time.time()-s)

