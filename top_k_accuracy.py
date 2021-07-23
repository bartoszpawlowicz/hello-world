import pickle
import time
import keras
import numpy as np
from sklearn.neighbors import KDTree
import mlflow


def dump_embeddings(x, model) -> np.ndarray:
    embeddings = []
    for sample in x:
        embedding = model.predict(np.expand_dims(sample, axis=0))
        embeddings.append(embedding)
    return np.vstack(embeddings)


def accuracy_at_k(y_true: np.ndarray, embeddings: np.ndarray, k: int, sample: int = None) -> float:
    kdtree = KDTree(embeddings)
    if sample is None:
        sample = len(y_true)
    y_true_sample = y_true[:sample]

    indices_of_neighbours = kdtree.query(embeddings[:sample], k=k + 1, return_distance=False)[:, 1:]

    y_hat = y_true[indices_of_neighbours]

    matching_category_mask = np.expand_dims(np.array(y_true_sample), -1) == y_hat

    matching_cnt = np.sum(matching_category_mask.sum(-1) > 0)
    accuracy = matching_cnt / len(y_true_sample)
    return accuracy


def evaluate(dataset, embedding_model):
    [x, y_true] = dataset
    embeddings = dump_embeddings(x, embedding_model)
    accuracies = []
    for K in [1, 5, 10]:
        acc_k = accuracy_at_k(np.array(y_true), embeddings, K, 200)
        accuracies.append(acc_k)
        print(K, acc_k)
    return accuracies


class TopKCallback(keras.callbacks.Callback):
    def __init__(self, valid_pickle_path, period=1, top_k=None, name="valid", use_mlflow=True, **kwargs):
        if top_k is None:
            top_k = [1, 5, 10]
        self.top_k = top_k
        self.period = period
        self.valid_pickle_path = valid_pickle_path
        self.name = name
        self.use_mlflow = use_mlflow
        super(TopKCallback, self).__init__()
        self.__dict__.update(kwargs)

    def main_function(self, logs_dict=None):
        if logs_dict is None:
            logs_dict = {}
        start = time.time()
        with open(self.valid_pickle_path, 'rb') as f:
            validation_set = pickle.load(f)
            accuracies = evaluate(validation_set, self.model)
            for i in range(len(self.top_k)):
                logs_dict["top_K_" + self.name + "/top_" + str(self.top_k[i])] = float(accuracies[i])
        end = time.time()
        print(self.name + " top K checkpoint time: " + str(end - start))
        return logs_dict

    def on_epoch_end(self, epoch, logs=None, force=False):
        if self.period > 0:
            if epoch % self.period != 0:
                return
            logs.update(self.main_function())

    def on_train_begin(self, logs=None):
        accuracies = self.main_function()
        # logs.update(accuracies)
        print("Baseline top K before training:")
        print(accuracies)
        if self.use_mlflow:
            mlflow.log_text(str(accuracies), "baseline_top_K_" + self.name + ".txt")
            mlflow.log_metrics(accuracies, step=-1)
