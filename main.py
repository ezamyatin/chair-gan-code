import glob
import os
import sys
import time

from tqdm import tqdm

from model import Model, ModelSplit, ModelWOEMM
import argparse
import numpy as np
from scipy.misc import imsave
np.set_printoptions(suppress=True)

from dataset import H5PYDatasetAugRand


def main(args):
    print('compile')
    start = time.time()
    model = ModelSplit(args.reg)
    print('compiled in %1.3lf' % (time.time() - start))

    if args.weights is not None:
        model.load(args.weights)
        print('Model loaded from', args.weights)

    if not os.path.exists(args.output + '/samples'): os.makedirs(args.output + '/samples')

    init_counter = args.batch_count * args.start_epoch
    dataset = H5PYDatasetAugRand(args.train_h5,
                                 args.batch_size,
                                 args.gamma,
                                 init_counter,
                                 args.batch_count,
                                 True,
                                 args.aug_inc)

    start_epoch = args.start_epoch
    print('start train from epoch', start_epoch)
    iter_count = 0
    for epoch in range(start_epoch, args.epoch_count):
        start = time.time()
        disc_count, disc_err = 0, 0
        gen_count, gen_err = 0, 0
        time_in_train = 0
        for batch in tqdm(dataset(args.batch_size), total=args.batch_count):
            iter_count += 1
            t = time.time()
            d_err = model.train_disc(batch['image'], batch['label'], batch['angle'], batch['t'])
            time_in_train += time.time() - t

            disc_err = d_err + disc_err
            disc_count += 1
            if iter_count % args.d_to_g != 0:
                continue
            t = time.time()
            g_err = model.train_gen(batch['label'], batch['angle'], batch['t'])
            time_in_train += time.time() - t
            gen_err = g_err + gen_err
            gen_count += 1

        disc_err /= disc_count
        gen_err /= gen_count

        print('epoch:', epoch)
        print('disc_err:', disc_err)
        print('gen_err:', gen_err)
        print('time: %1.3lf' % (time.time() - start))
        print('time in train: % 1.3lf' % time_in_train)
        model.save(args.output + '/model.pkl')

        batch = next(dataset(16, parallelize=False))
        samples = batch['image']
        samples = samples.reshape((4, 4, 64, 64, 4))
        samples = np.concatenate(samples, axis=1)
        samples = np.concatenate(samples, axis=1)
        imsave(args.output + '/samples/%04d_data.png' % epoch, samples)

        samples = model.gen(batch['label'], batch['angle'], batch['t'])
        samples = samples.reshape((4, 4, 64, 64, 4))
        samples = np.concatenate(samples, axis=1)
        samples = np.concatenate(samples, axis=1)
        imsave(args.output + '/samples/%04d_gen.png' % epoch, samples)


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_arg_parser.add_argument('--train-h5', '-t', type=str, required=True, help='Path to h5 train file')
    main_arg_parser.add_argument('--weights', '-w', type=str, default=None, help='Path to model weights')
    main_arg_parser.add_argument('--output', '-o', type=str, required=True, help='Path to output dir')
    main_arg_parser.add_argument('--batch-count', '-bc', type=int, default=843 * 62 // 16,
                                 help='Number of batches in one epoch')
    main_arg_parser.add_argument('--epoch-count', '-ec', type=int, default=1000,
                                 help='Number of epochs')
    main_arg_parser.add_argument('--start-epoch', '-se', type=int, default=0,
                                 help='Start epoch. (Influence to augmentation)')
    main_arg_parser.add_argument('--batch-size', '-bs', type=int, default=16,
                                 help='Count images in batch')
    main_arg_parser.add_argument('--gamma', type=float, default=5e-5,
                                 help='Gamma from augmentation')
    main_arg_parser.add_argument('--aug-inc', type=float, default=1.,
                                 help='Inc from augmentation')
    main_arg_parser.add_argument('--d-to-g', type=int, default=3,
                                 help='D per G trains')
    main_arg_parser.add_argument('--reg', action='store_true',
                                 help='Use regularization')

    args = main_arg_parser.parse_args()
    main(args)
