from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import random
import cv2
import os
from scipy import ndimage
from itertools import islice
import pandas as pd


def add_noise(img, std):
    mean = 0.0  # some constant
    # std = 50.0  # some constant (standard deviation)  # range(0,50)
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise
    noisy_img_clipped = np.ndarray.astype(noisy_img_clipped, int)
    return noisy_img_clipped


def change_hue_and_saturation(img, hue_change, sat_change):
    """
    :param img: image for augmentation
    :param hue_change: hue change, reasonable range(-15,15)
    :param sat_change: saturation change, reasonable range(0, 2)
    :return: augmented image
    """
    try:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # sprawdzić zamianę kolorów
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
        (h, s, v) = cv2.split(hsv_img)
        s = s * sat_change
        h = h + hue_change
        s = np.clip(s, 0, 255)
        h = np.clip(h, 0, 255)
        augm_hsv_img = cv2.merge([h, s, v])
        augm_img = cv2.cvtColor(augm_hsv_img.astype("uint8"), cv2.COLOR_HSV2RGB)
        return augm_img
    except cv2.error as e:
        print(e)


def blur_image(img, sigma):
    blurred_img = ndimage.gaussian_filter(img, sigma)  # sigma range(0,2)
    return blurred_img


class TupletBatchGenerator(Sequence):
    def __init__(self, images_paths, forth_same, batch_size=16, shuffle=True, img_shape=(160, 160, 3), do_augm=True,
                 hardest_df_path=None):

        self.forth_same = forth_same    # True or False or None (for triplet)
        self.tuplet_type = 4 if self.forth_same is not None else 3

        self.images_paths = list(images_paths.values())

        self.total_size = sum([len(x) for x in self.images_paths])

        self.image_index = []
        self.path_to_image_index = {}
        self.hardest_df_path = hardest_df_path
        self.hardest_df = None
        self.create_image_index()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_shape = img_shape
        self.do_augm = do_augm
        self.datagen = ImageDataGenerator(
                    # rescale=1. / 255,
                    # featurewise_center=False,
                    # featurewise_std_normalization=False,
                    # channel_shift_range=64,
                    brightness_range=(0.7, 1.3),
                    shear_range=10,
                    rotation_range=90,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=[0.8, 1.2],
                    fill_mode='constant',
                    cval=0,
                    vertical_flip=True,
                    horizontal_flip=True,
        )

        if self.shuffle:
            self.on_epoch_end()

    def create_image_index(self):
        self.image_index = []
        for i, class_images_list in enumerate(self.images_paths):
            for j, img_path in enumerate(class_images_list):
                self.image_index.append((i, j))
                self.path_to_image_index[img_path] = (i, j)
        if self.hardest_df_path is not None:
            if os.path.exists(self.hardest_df_path):
                print('loading hardest df')
                self.hardest_df = pd.read_pickle(self.hardest_df_path)

    def __len__(self):
        return int(np.ceil(float(self.total_size)/self.batch_size))

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > self.total_size:
            r_bound = self.total_size
            l_bound = r_bound - self.batch_size

        no_anchor_images = r_bound-l_bound

        instance_count = 0

        x1_batch = np.zeros((no_anchor_images * self.tuplet_type,) + self.img_shape)

        y_batch = np.zeros((no_anchor_images * self.tuplet_type,), dtype=int)

        for example_index in self.image_index[l_bound:r_bound]:
            class_i, image_i = example_index

            img_path_selected = self.images_paths[class_i][image_i]
            x1_batch[instance_count] = self.get_image(img_path_selected)
            y_batch[instance_count] = int(class_i)

            image_index_pool = [x for x in range(len(self.images_paths[class_i])) if x != image_i]
            x1_batch[instance_count + 1] = self.get_image(self.images_paths[class_i][random.choice(image_index_pool)])
            y_batch[instance_count + 1] = int(class_i)

            other_class_index_pool = [x for x in range(len(self.images_paths)) if x != class_i]
            if self.hardest_df is not None:
                hardest_path = self.hardest_df.loc[self.hardest_df['img_path'] == img_path_selected]['hardest_negative'].iloc[0]
                other_class_i, n_i = self.path_to_image_index[hardest_path]
            else:
                other_class_i = random.choice(other_class_index_pool)
                image_index_pool = [x for x in range(len(self.images_paths[other_class_i]))]
                n_i = random.choice(image_index_pool)
            x1_batch[instance_count + 2] = self.get_image(self.images_paths[other_class_i][n_i])
            y_batch[instance_count + 2] = int(other_class_i)

            if self.tuplet_type == 4:
                if not self.forth_same:
                    other_class_i = random.choice(other_class_index_pool)
                image_index_pool = [x for x in range(len(self.images_paths[other_class_i]))]
                image_index_pool.remove(n_i)
                n_i_2 = random.choice(image_index_pool)
                x1_batch[instance_count + 3] = self.get_image(self.images_paths[other_class_i][n_i_2])
                y_batch[instance_count + 3] = int(other_class_i)

            instance_count += self.tuplet_type

        return x1_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images_paths)

            self.create_image_index()

    def get_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_shape[0], self.img_shape[1]))
        if self.do_augm:
            hue_diff = random.randint(-5, 5)
            sat_change = random.uniform(0.8, 1.2)
            image = change_hue_and_saturation(image, hue_diff, sat_change)
            # image = change_hue_and_saturation(image, 1, 1)
            image = self.datagen.random_transform(image)
            # image = np.random.uniform(image, )
            noise_std = random.randint(0, 30)
            image = add_noise(image, noise_std)
            blur_sigma = random.uniform(0, 0.2)
            image = blur_image(image, blur_sigma)
        image = image / 128 - 1

        return image


class TripletBatchGenerator(TupletBatchGenerator):
    def __init__(self, images_paths, batch_size=16, shuffle=True, img_shape=(160, 160, 3), do_augm=True):
        super().__init__(images_paths, forth_same=None, batch_size=batch_size,
                         shuffle=shuffle, img_shape=img_shape, do_augm=do_augm)

