import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt


# get the face embedding for one face
def get_embedding_facenet(model, input_filepath):
    image = Image.open(input_filepath)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    face_pixels = np.asarray(image)
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def get_embedding_from_path(path, nn_model, input_shape, show_img=False):
    image = cv2.imread(path)
    image_embedding = get_embedding(image, nn_model, input_shape, show_img=show_img)
    return image_embedding


def get_embedding(image, nn_model, input_shape, show_img=False):
    image = cv2.resize(image, input_shape[0:2])
    # image=image[:,:,::-1]
    draw = image
    image = image / 128 - 1
    img_input_shape = (1,) + input_shape
    image_embedding = nn_model.predict(image.reshape(img_input_shape))
    if show_img:
        # draw = draw[:,:,::-1]
        plt.imshow(draw)
        plt.show()
    return image_embedding


def return_distances_as_array(labels, embeddings, squared=False):
    """
    Computing distances between all possible embeddings and then return them as separate arrays
    for positive pairs (same labels) and negative pairs (different labels)
    :param labels: class id vector for each embedding
    :param embeddings: 2D tensor of embeddings
    :param squared: bool if distance should be squared or not (see distance function)
    :return: 4 arrays that contain distances and indexes
    """
    distances = _pairwise_distances_np(embeddings, squared=squared)
    positive_distances = np.array([])
    negative_distances = np.array([])
    a_p_idxs = []
    a_n_idxs = []
    positive_mask = _get_anchor_positive_mask_np(labels)
    positive_mask = _remove_duplicates_from_mask_np(positive_mask)
    negative_mask = _get_anchor_negative_mask_np(labels)
    negative_mask = _remove_duplicates_from_mask_np(negative_mask)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if positive_mask[i, j]:
                positive_distances = np.append(positive_distances, distances[i, j])
                a_p_idxs.append([i, j])
            if negative_mask[i, j]:
                negative_distances = np.append(negative_distances, distances[i, j])
                a_n_idxs.append([i, j])
    return positive_distances, negative_distances, a_p_idxs, a_n_idxs


def _get_anchor_positive_mask_np(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: Array with shape [batch_size]
    Returns:
        mask: Numpy Array with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = np.eye(np.shape(labels)[0])
    logical_equal = indices_equal == 1
    indices_not_equal = np.logical_not(logical_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))

    # Combine the two masks
    mask = np.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_mask_np(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: Array with shape [batch_size]
    Returns:
        mask: Numpy Array with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))

    mask = np.logical_not(labels_equal)

    return mask


def _remove_duplicates_from_mask_np(mask):
    over_diagonal_zeros = np.tril(np.ones(np.shape(mask)[0]))
    over_diagonal_false = np.logical_not(over_diagonal_zeros)
    fixed_mask = np.logical_and(over_diagonal_false, mask)
    return fixed_mask


def _pairwise_distances_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
            pairwise_distances.diagonal())
    return pairwise_distances


def get_all_distances_np(labels_list, embeddings_list):
    positive_distances = []
    negative_distances = []

    labels_array = np.asarray(labels_list)
    embeddings_array = np.asarray(embeddings_list)

    print(labels_array.shape)
    print(embeddings_array.shape)

    p_distances, n_distances, a_p_ids, a_n_ids = return_distances_as_array(labels_array, embeddings_array, squared=False)
    for d, distance in enumerate(p_distances):
        positive_distances.append(distance)
    for d, distance in enumerate(n_distances):
        negative_distances.append(distance)

    return positive_distances, negative_distances


def compute_eer(positive_distances, negative_distances):
    data = []
    label = []
    for distance in positive_distances:
        data.append(distance)
        label.append(0)
    for distance in negative_distances:
        data.append(distance)
        label.append(1)

    fpr, tpr, thresholds = roc_curve(label, data, pos_label=1)
    auc_score = auc(fpr, tpr)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh, auc_score
