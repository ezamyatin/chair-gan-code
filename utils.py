import pickle
import numpy as np
import time
import os


def save(file, obj):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w+b') as out:
        pickle.dump(obj, out)


def load(file):
    with open(file, 'rb') as data:
        return pickle.load(data)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('time: %1.3lfs' % (te - ts))
        return result

    return timed


def one_hot(x, n):
    if not hasattr(x, "__len__") or len(x) == 1:
        res = np.zeros(n, dtype=np.float32)
        res[x] = 1
        return res
    res = np.zeros((len(x), n), dtype=np.float32)
    res[np.arange(len(x)), x] = 1
    return res


def sincos2angle(angle):
    angle = np.array(angle)
    if len(angle.shape) == 1:
        return np.array([np.around(np.rad2deg(np.array(np.arctan2(angle[0], angle[1])))),
                         np.around(np.rad2deg(np.array(np.arctan2(angle[2], angle[3]))))])
    return np.array([np.around(np.rad2deg(np.array(np.arctan2(angle[:, 0], angle[:, 1])))),
                     np.around(np.rad2deg(np.array(np.arctan2(angle[:, 2], angle[:, 3]))))]).transpose()


def angle2sincos(angle):
    angle = np.array(angle)
    angle = np.deg2rad(angle)
    if len(angle.shape) == 1:
        return np.array([np.sin(angle[0]), np.cos(angle[0]),
                         np.sin(angle[1]), np.cos(angle[1])])
    return np.array([np.sin(angle[:, 0]), np.cos(angle[:, 0]),
                     np.sin(angle[:, 1]), np.cos(angle[:, 1])]).transpose()