import argparse
import sys

import h5py
import numpy as np
from scipy.misc import imread, imsave
import os

from tqdm import tqdm

from utils import one_hot, sincos2angle, angle2sincos
from model import ModelSplit

# DEFAULT_T = np.float32([0, 0, 0, 1, 1, 0, 1, 1])
# DEFAULT_ANGLE = np.float32([0.34202015, 0.93969262, -0.89100653, -0.45399049])

# ANGLES1 = [20.0, 30.0]
# ANGLES2 = [-175.0, -163.0, -151.0, -140.0, -128.0, -117.0, -105.0, -93.0, -82.0, -70.0, -59.0, -47.0, -35.0, -24.0,
#           -12.0, 0.0, 11.0, 23.0, 34.0, 46.0, 58.0, 69.0, 81.0, 92.0, 104.0, 116.0, 127.0, 139.0, 150.0, 162.0, 174.0]
# ANGLES = [(i, j) for i in ANGLES1 for j in ANGLES2]


# def morph(model, c1, c2, angle=DEFAULT_ANGLE, count=10):
#    gen_frames = model.gen({'label': one_hot([c1] * count, 843) * (1 - np.arange(count)[:, np.newaxis] / count) + \
#                                     one_hot([c2] * count, 843) * np.arange(count)[:, np.newaxis] / count,
#                            'angle': np.array([angle] * count),
#                            't': DEFAULT_T[np.newaxis].repeat(count, axis=0)})
#    return gen_frames


# def gen(model, c, v, t=DEFAULT_T):
#    gen_frames = model.gen({'label': one_hot([c] * 2, 843),
#                            'angle': np.array([v] * 2),
#                            't': t[np.newaxis].repeat(2, axis=0)})
#    return gen_frames[0]

K = 30


def main(args):
    model = ModelSplit(reg=False)
    model.load(args.root)
    np.random.seed(239)
    dataset = h5py.File('./data/chairs_64x64.h5', 'r')
    c = np.random.choice(np.arange(843), K)
    v_idx = np.random.choice(np.arange(len(dataset['angle']), K))
    v = dataset['angle'][v_idx]
    t = np.float32([0, 0, 0, 1, 1, 0, 1, 1])[np.newaxis].repeat(K, axis=0)
    result = model.gen(c, v, t)
    if not os.path.exists(args.output): os.makedirs(args.output)
    for i, img in enumerate(result):
        imsave(args.output + '/%d.png' % i, img)


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser()
    main_arg_parser = argparse.ArgumentParser()
    main_arg_parser.add_argument("--root", type=str, required=True, help='Path to model root')
    main_arg_parser.add_argument("--output", "-o", type=str, required=True, help='Path to output')

    args = main_arg_parser.parse_args()
    main(args)
