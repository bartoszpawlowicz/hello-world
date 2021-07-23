import keras
import time
import pandas as pd
import pickle
from tqdm import tqdm
import cv2
import numpy as np
import mlflow
import os

EMBEDDING = 'embedding'
IMG_PATH = 'img_path'
HARDEST_NEGATIVE = 'hardest_negative'
NAME = 'product_name'


class OfflineMining(keras.callbacks.Callback):
    def __init__(self, images_paths, df_path, input_shape, period=1):
        super().__init__()
        self.images_paths = images_paths
        self.df_path = df_path
        self.period = period
        self.input_shape = input_shape

        imgs_paths_dict = self.images_paths

        paths_list = []
        keys_list = []
        for k in imgs_paths_dict.keys():
            for elem in imgs_paths_dict[k]:
                paths_list.append(elem)
                keys_list.append(k)

        df_data = {IMG_PATH: paths_list, NAME: keys_list}
        df = pd.DataFrame(data=df_data)

        df.to_pickle(self.df_path)

    def on_train_begin(self, logs=None):
        update_embeddings(self.df_path, self.model, self.input_shape)

    def on_epoch_end(self, epoch, logs=None):
        update_embeddings(self.df_path, self.model, self.input_shape)


def update_embeddings(df_path, nn_model, input_shape):
    df = pd.read_pickle(df_path)

    tqdm.pandas(desc="computing embeddings: ")
    df[EMBEDDING] = df[IMG_PATH].progress_apply(
        lambda s: get_embedding_from_path(s, nn_model, input_shape))

    tqdm.pandas(desc="finding hardest negative: ")
    df[HARDEST_NEGATIVE] = df.progress_apply(lambda s: find_hardest(s, df.copy()), axis=1)

    df.to_pickle(df_path)
    df.to_csv(df_path.replace(".p", ".txt"))


def find_hardest(s, df):
    q_embedding = s[EMBEDDING]
    df.loc[df[NAME] != s[NAME], 'SIMILARITY'] = df[EMBEDDING].apply(lambda p: np.dot(q_embedding[0], p[0]))
    best = df[df['SIMILARITY'] == df['SIMILARITY'].max()].iloc[0]
    return best[IMG_PATH]


def get_embedding_from_path(path, nn_model, input_shape):
    image = cv2.imread(path)
    image_embedding = get_embedding(image, nn_model, input_shape)
    return image_embedding


def get_embedding(image, nn_model, input_shape, show_img=False):
    image = cv2.resize(image, input_shape[0:2])
    image = image / 128 - 1
    img_input_shape = (1,) + input_shape
    image_embedding = nn_model.predict(image.reshape(img_input_shape))
    return image_embedding

