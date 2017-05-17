import numpy as np
import skimage.transform as tf
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import hsv_to_rgb


class SpatialAugmentator:
    def __init__(self, rotation=0, translation=(0, 0), stretch=(1, 1)):
        stretch = np.array(stretch)
        self.params = np.array([rotation, translation[0], translation[1],
                                stretch[0], stretch[1]], dtype=np.float32)
        self.transforms = []
        self.transforms.append(
            lambda x, c: tf.warp(x, tf.SimilarityTransform(translation=np.int32((1 - stretch) * x.shape[0] / 2),
                                                           scale=stretch),
                                 output_shape=x.shape, cval=c, preserve_range=True))
        self.transforms.append(lambda x, c: tf.rotate(x, np.rad2deg(rotation), cval=c, preserve_range=True))
        self.transforms.append(
            lambda x, c: tf.warp(x, tf.SimilarityTransform(translation=np.int32(translation * x.shape[0])),
                                 output_shape=x.shape, cval=c, preserve_range=True))

    def __call__(self, img, cval):
        for t in self.transforms:
            img = t(img, cval)
        return img


class ChromaticAugmentation:
    def __init__(self, col_angle=0, sat_mul=1, br_mul=1):
        self.col_angle = col_angle
        self.sat_mul = sat_mul
        self.br_mul = br_mul
        self.params = np.array([col_angle, sat_mul, br_mul], dtype=np.float32)

    def __call__(self, img):
        img = np.float32(img)
        hsv = rgb_to_hsv(img)
        hsv[:,:,0] += np.rad2deg(self.col_angle % (2 * np.pi)) / 360
        hsv[:,:,0] %= 1
        hsv[:,:,1] *= self.sat_mul
        hsv[:,:,2] *= self.br_mul
        hsv[hsv > 1] = 1
        img = hsv_to_rgb(hsv)
        return img



def compute_discount(num_iter, initial_coeff=0.1, final_coeff=1., gamma=1e-3):
    return initial_coeff + (final_coeff - initial_coeff) * \
                           (2 / (1 + np.exp(-gamma * num_iter)) - 1)


def uniform_bernoulli(l, r, p, size=None, exp=False, discount=1.):
    m = (l + r) / 2
    spread = r - m
    spread *= discount
    l, r = m - spread, m + spread
    t = np.random.binomial(1, p, size) * np.random.uniform(l, r, size)
    if exp:
        t = np.exp(t)
    return t


def perform_augmentation(sample, counter, params=False, gamma=2e-3):
    discount = compute_discount(counter, gamma=gamma)
    bernoulli_discount = compute_discount(counter, 0.5, 1., gamma)
    img = sample[..., :3]
    seg = sample[..., 3:].repeat(3, axis=-1)

    # Changed spatial random borders and bernoulli p
    c_augmentator = ChromaticAugmentation(col_angle=uniform_bernoulli(-1.5, 1.5, bernoulli_discount, discount=discount),
                                          sat_mul=uniform_bernoulli(-0.7, 0.7, bernoulli_discount, exp=True,
                                                                    discount=discount),
                                          br_mul=uniform_bernoulli(-0.7, 0.7, bernoulli_discount, exp=True,
                                                                   discount=discount))

    s_augmentator = SpatialAugmentator(rotation=uniform_bernoulli(-0.1, 0.1, bernoulli_discount, discount=discount),
                                       translation=uniform_bernoulli(-0.05, 0.05, bernoulli_discount, size=2,
                                                                     discount=discount),
                                       stretch=uniform_bernoulli(-0.1, 0.1, bernoulli_discount, size=2, exp=True,
                                                                 discount=discount))
    img = c_augmentator(img)
    img = s_augmentator(img, 0)
    seg = s_augmentator(seg, 0)
    seg = seg.mean(axis=-1, keepdims=True)
    sample = np.concatenate([img, seg], axis=-1)
    if not params:
        return sample
    return sample, np.hstack((s_augmentator.params, c_augmentator.params))
