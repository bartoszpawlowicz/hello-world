import numpy as np
import tensorflow as tf
from vsearch.utils.distances_utils import pairwise_distance as get_distances
from vsearch.utils.distances_utils import pairwise_distances_np


dtype = tf.float32


def compute_alfa_np(distances, mask, n_mask):
    p_dist = distances * mask
    p_mean = p_dist.sum() / np.clip(mask.sum(), a_min=1e-12, a_max=None)
    n_dist = distances * n_mask
    n_mean = n_dist.sum() / np.clip(n_mask.sum(), a_min=1e-12, a_max=None)

    return n_mean - p_mean


def compute_alfa_tf(distances, mask, n_mask):
    p_dist = distances * mask
    p_mean = tf.math.reduce_sum(p_dist) / tf.clip_by_value(tf.math.reduce_sum(mask), clip_value_min=1e-12,
                                                           clip_value_max=dtype.max)
    n_dist = distances * n_mask
    n_mean = tf.math.reduce_sum(n_dist) / tf.clip_by_value(tf.math.reduce_sum(n_mask), clip_value_min=1e-12,
                                                           clip_value_max=dtype.max)

    return n_mean - p_mean


def get_masks_np(labels):
    n = np.shape(labels)[0]
    mask = np.equal(np.expand_dims(labels, axis=0), np.expand_dims(labels, axis=1))
    n_mask = ~mask
    n_mask = np.array(n_mask, dtype=float)
    mask = np.array(mask, dtype=float)
    mask = mask - np.eye(n)
    return mask, n_mask


def get_masks_tf(labels):
    n = tf.shape(labels)[0]
    mask = tf.equal(tf.expand_dims(labels, axis=0), tf.expand_dims(labels, axis=1))
    n_mask = ~mask
    n_mask = tf.cast(n_mask, dtype=dtype)
    mask = tf.cast(mask, dtype=dtype)
    # mask = mask - np.eye(n)
    return mask, n_mask


def l_1_np(distances, mask, n_mask, alfa_1=0):
    p_n_mask_3d = np.matmul(np.expand_dims(mask, axis=-1), np.expand_dims(n_mask, axis=1))
    dist_3d = np.clip(np.expand_dims(distances, axis=-1) - np.expand_dims(distances, axis=1) + alfa_1, a_min=0,
                      a_max=None)
    L1 = (dist_3d * p_n_mask_3d).sum()
    return L1


def l_1_tf(distances, mask, n_mask, alfa_1=0):
    p_n_mask_3d = tf.matmul(tf.expand_dims(mask, axis=-1), tf.expand_dims(n_mask, axis=1))
    dist_3d = tf.clip_by_value(tf.expand_dims(distances, axis=-1) - tf.expand_dims(distances, axis=1) + alfa_1,
                               clip_value_min=0, clip_value_max=dtype.max)

    L_1 = tf.math.reduce_sum(dist_3d * p_n_mask_3d)
    return L_1


def l_2_np(distances, mask, n_mask, alfa_2=0):
    n_mask_3d = np.matmul(np.expand_dims(n_mask, axis=2), np.expand_dims(n_mask, axis=1))
    dist_4d = np.clip(
        np.expand_dims(distances, axis=0) - np.expand_dims(np.expand_dims(distances, axis=-1), axis=-1) + alfa_2,
        a_min=0, a_max=None)
    L2 = (
            np.transpose(dist_4d, (2, 1, 0, 3)) * np.expand_dims(n_mask_3d, axis=-1) * np.expand_dims(
        np.expand_dims(n_mask, axis=0), axis=-1) * np.expand_dims(np.expand_dims(mask, axis=1), axis=1)
    ).sum()
    return L2


def l_2_tf(distances, mask, n_mask, alfa_2=0):
    n_mask_t2 = tf.expand_dims(n_mask, axis=2)
    n_mask_t1 = tf.expand_dims(n_mask, axis=1)
    n_mask_3d = tf.matmul(n_mask_t2, n_mask_t1)
    dist_4d = tf.clip_by_value(
        tf.expand_dims(distances, axis=0) - tf.expand_dims(tf.expand_dims(distances, axis=-1), axis=-1) + alfa_2,
        clip_value_min=0, clip_value_max=dtype.max)
    L2 = tf.math.reduce_sum(
        tf.transpose(dist_4d, (2, 1, 0, 3)) *
        tf.expand_dims(n_mask_3d, axis=-1) *
        tf.expand_dims(tf.expand_dims(n_mask, axis=0), axis=-1) *
        tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
    )
    return L2


def quad_loss_param_np(margin=None, squared=False):
    def quad_loss_np(labels, x):
        distances = pairwise_distances_np(x, squared=squared)
        mask, n_mask = get_masks_np(labels)

        if margin is None:
            alfa_1 = compute_alfa_np(distances, mask, n_mask)
        else:
            alfa_1 = margin
        alfa_2 = 0.5 * alfa_1

        L_1 = l_1_np(distances, mask, n_mask, alfa_1=alfa_1)
        L_2 = l_2_np(distances, mask, n_mask, alfa_2=alfa_2)

        loss = L_1 + L_2
        return loss

    return quad_loss_np


def quad_loss_param(margin=None, squared=False):
    def quad_loss(labels, x):
        labels = tf.squeeze(labels)
        # distances = pairwise_distance(x, squared=squared)
        distances = get_distances(x, squared=squared)
        mask, n_mask = get_masks_tf(labels)

        if margin is None:
            alfa_1 = compute_alfa_tf(distances, mask, n_mask)
        else:
            alfa_1 = margin
        alfa_2 = 0.5 * alfa_1

        L_1 = l_1_tf(distances, mask, n_mask, alfa_1=alfa_1)
        # L_2 = 0
        L_2 = l_2_tf(distances, mask, n_mask, alfa_2=alfa_2)

        loss = L_1 + L_2
        return loss
    return quad_loss


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # ta = tf.constant([[1, 0], [-1, 0]], dtype=tf.float32)
    # v = pairwise_distance(ta, squared=True)
    # with tf.Session() as sess:
    #     print(v.eval())

    # print(_pairwise_distances_np(np.array([[1, 0], [-1, 0]], dtype=np.float64), squared=True))
    # print(_pairwise_distances(np.array([[1, 0], [-1, 0]], dtype=np.float64), squared=True))
    # print(pairwise_distance(np.array([[1, 0], [-1, 0]], dtype=np.float), squared=True))

    # xx = np.array(
    #     [[1, 0], [-1, 0], [-1, -1], [-1, -1], [-1, -1]]
    #     , dtype=np.float
    # )
    # yy = np.array(
    #     [[1], [1], [2], [3], [3]]
    #     , dtype=np.float
    # )
    #
    # q_l_np = quad_loss_param_np(margin=0, squared=True)
    # print(q_l_np(yy, xx))

    xx = tf.convert_to_tensor(
        [[1, 0], [-1, 0], [-1, -1], [-1, -1], [-1, -1]]
        , dtype=float
    )
    yy = tf.convert_to_tensor(
        # [[1, 1, 2, 3, 3]]
        [[1], [1], [2], [3], [3]]
        , dtype=float
    )
    print(yy.shape)
    print(xx.shape)

    q_l = quad_loss_param(margin=0, squared=False)
    print(q_l(yy, xx))
