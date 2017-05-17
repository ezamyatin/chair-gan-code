import random
from multiprocessing import Value
import os
import h5py
import numpy as np
from multiprocessing import Queue
from multiprocessing import Process
from augmentation import perform_augmentation


def prepare(batch):
    batch['image'] = np.float32(batch['image'] / 255)
    batch['mask'] = np.float32(batch['mask'] / 255)
    batch['image'] *= batch['mask']
    batch['image'] = np.concatenate((batch['image'],
                                     batch['mask'].mean(axis=-1, keepdims=True)),
                                    axis=-1)
    del batch['mask']
    return batch


def do_augmentation(batch, counter, gamma):
    image = batch['image']
    result = []
    teta = []
    for sample in image:
        i, t = perform_augmentation(sample, counter, True, gamma)
        result.append(i)
        teta.append(t)
    result = np.concatenate([i[np.newaxis] for i in result])
    teta = np.concatenate([i[np.newaxis] for i in teta])
    batch['image'] = result
    batch['t'] = teta
    return batch


class H5PYDatasetAugRand:
    def __init__(self, h5_file, max_batch_size, gamma, init_counter=0, call_len=-1, aug=True, counter_inc=1.):
        self.batch_q = Queue(100)
        self.shared_counter = Value('f')
        self.shared_counter.value = init_counter
        self.max_batch_size = max_batch_size
        self.call_len = call_len
        self.gamma = gamma

        def target_f(batch_q, shared_counter):
            db = h5py.File(h5_file, 'r')
            n = len(db['label'])
            np.random.seed(os.getpid())
            while True:
                idx = sorted(random.sample(range(0, n), max_batch_size))
                batch = {k: v[idx] for k, v in db.items()}
                batch = prepare(batch)
                if aug:
                    batch = do_augmentation(batch, shared_counter.value, gamma)
                batch_q.put(batch)
                shared_counter.value += counter_inc
        self.pool = [Process(target=target_f, args=(self.batch_q, self.shared_counter)) for i in range(5)]
        for p in self.pool:
            p.start()

    def __call__(self, batchsize, parallelize=True):
        c = 0
        while True:
            batch = self.batch_q.get()
            assert self.max_batch_size >= batchsize
            batch = {k: v[:batchsize] for k, v in batch.items()}
            yield batch
            c += 1
            if c == self.call_len: break