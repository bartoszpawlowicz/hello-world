import os
import numpy as np
from PIL import Image
import functools


def image_transpose_exif(im):
    """
        Apply Image.transpose to ensure 0th row of pixels is at the visual
        top of the image, and 0th column is the visual left-hand side.
        Return the original image if unable to determine the orientation.

        As per CIPA DC-008-2012, the orientation field contains an integer,
        1 through 8. Other values are reserved.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)


def get_filepaths(folder_path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            f.append(path)
    return f


def fit_img_from_path_into_square(img_path, cval=0):
    img = Image.open(img_path)
    bcg = fit_img_into_square(img, cval=cval)
    return bcg


def fit_img_into_square(img, cval=0):
    width, height = img.size
    size = (max(width, height), max(width, height))
    img.thumbnail(size, Image.ANTIALIAS)
    bcg = Image.new('RGB', size, (cval, cval, cval))
    bcg.paste(img,
              ((size[0] - img.size[0]) // 2,
               (size[1] - img.size[1]) // 2))
    return bcg


def fit_array_into_square(array):
    img = Image.fromarray(array)
    cropped = fit_img_into_square(img)
    new_array = np.array(cropped)
    return new_array


def reverse_fit_img_from_path_into_square(img_path, cval=0):
    img = Image.open(img_path)
    bcg = reverse_fit_img_into_square(img, cval=cval)
    return bcg


def reverse_fit_img_into_square(img, cval=0):
    img_array = np.array(img)

    for i in range(img_array.shape[0]):
        if not (np.sum(img_array[i, :, :] != cval) == 0):
            img_array = img_array[i::, :, :]
            break

    for i in range(img_array.shape[0]-1, 0, -1):
        if not (np.sum(img_array[i, :, :] != cval) == 0):
            img_array = img_array[0:i, :, :]
            break

    for i in range(img_array.shape[1]):
        if not (np.sum(img_array[:, i, :] != cval) == 0):
            img_array = img_array[:, i::, :]
            break

    for i in range(img_array.shape[1]-1, 0, -1):
        if not (np.sum(img_array[:, i, :] != cval) == 0):
            img_array = img_array[:, 0:i, :]
            break
    return Image.fromarray(img_array)
