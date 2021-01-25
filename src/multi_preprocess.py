import multiprocessing
import time
import numpy as np
from h5data_reader import H5DataReader

starttime = time.time()


def func(data, factor):
    return data * factor


def handleData(func, data, args, queue):
    res = func(data, *args)
    queue.put(res)

class MultiProcessDataHandler(object):

    def __init__(self, default_func=None, default_func_args=[], processes=multiprocessing.cpu_count(), capacity=16, task=handleData):
        self.capacity = capacity
        self.processes = processes
        self.default_func = default_func
        self.default_func_args = default_func_args
        self.pool = multiprocessing.Pool(processes=processes)
        print(f"create {processes} processes for preprocess")
        self.queue = multiprocessing.Manager().Queue(maxsize=capacity)
        self.task = task

    def handle(self, data, func=None, args=None):
        if not func :
            func = self.default_func
        if not args :
            args = self.default_func_args
        return self.pool.apply_async(self.task, (func, data, args, self.queue))

    def get_result(self):
        return self.queue.get()

def plus(data,add):
    return data+add

if __name__ == '__main__':
    handler = MultiProcessDataHandler(default_func=func, default_func_args=(1,))
    myplus =plus
    handler.handle(2)
    handler.handle(3,myplus,(9,))
    handler.handle(4)
    print(handler.get_result())
    print(handler.get_result())
    print(handler.get_result())


    # pool = multiprocessing.Pool(processes=3)
    # queue = multiprocessing.Manager().Queue(maxsize=2)
    # reader = H5DataReader('../test.h5','r',data_key='signals')
    # print(reader.shuffle_indices)
    # result = []
    # lis = []
    # starttime = time.time()
    # e = pool.apply_async(foo,(1,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (2,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (3,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (4,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (5,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (6,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (7,queue,reader.get_shuffle_data(2)[1]))
    # pool.apply_async(foo, (8,queue,reader.get_shuffle_data(2)[1]))
    # time.sleep(4)
    # while True:
    #     print('get data,',queue.qsize(),queue.get())
    # for i in range(10):
    #     result.append(pool.apply_async(foo,(i,)))
    # pool.close()
    # pool.join()
    # print(time.time()-starttime)
    # for res in result:
    #     lis.append(res.get())
    # print(lis)
    # print(time.time()-starttime)

    # for i in range(10):
    #     result.append(pool.apply_async(lambda x: x +1,(i,)))
    # pool.close()
    # pool.join()
    # print(time.time()-starttime)
    # for res in result:
    #     lis.append(res.get())
    # print(lis)
    # print(time.time()-starttime)
