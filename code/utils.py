import os, random
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import confusion_matrix

EMB_PATH = os.path.join(os.path.dirname(__file__), '..', 'embeddings')
DS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')

USE_DAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder_4')  # USE with DAN architecture
USE_TRAN_PATH = os.path.join(EMB_PATH, 'universal-sentence-encoder-large_5')  # USE with Transformer architecture

DS_CLINC150_PATH = os.path.join(DS_PATH, 'CLINC150')
DS_ROSTD_PATH = os.path.join(DS_PATH, 'ROSTD')
DS_BANKING77_PATH = os.path.join(DS_PATH, 'BANKING77')


class Split:
    """
    Class used when splitting the training, validation and test set.

    :attributes:            intents_dct - keys: intent labels, values: unique ids, dict
                            new_key_value - keeps track of the newest unique id for intents_dct, int
                            embed_f - function that encodes sentences as embeddings
    """

    def __init__(self, embed_f):
        self.intents_dct = {}
        self.new_key_value = 0
        self.embed_f = embed_f

    def get_X_y(self, lst, limit_num_sents, set_type: str):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.

        :param:             lst - contains the dataset, list
                            limit_num_sents - specifies (if not None) the limited number of sentences per intent, int
                            set_type - deprecated; specifies the type of the received dataset (train, val or test), str
        :returns:           X - sentences encoded as embeddings, tf.Tensor OR sentences, list
                            y - intents, tf.Tensor
        """

        X = []
        y = []

        if limit_num_sents:  # these aren't needed normally
            random.shuffle(lst)
            label_occur_count = {}

        for sent, label in lst:
            if label not in self.intents_dct.keys():
                self.intents_dct[label] = self.new_key_value
                self.new_key_value += 1

            if limit_num_sents and label != 'oos':  # don't limit number of OOD sentences
                if label not in label_occur_count.keys():
                    label_occur_count[label] = 0

                if label_occur_count[label] == limit_num_sents:  # skip sentence and label if reached limit
                    continue

                label_occur_count[label] += 1

            X.append(sent)
            y.append(self.intents_dct[label])

        if self.embed_f is not None:
            X = self.embed_f(X)
            X = tf.convert_to_tensor(X, dtype='float32')

        y = tf.convert_to_tensor(y, dtype='int32')

        return X, y


def print_results(dataset_name: str, model_name: str, emb_name: str, results_dct: dict):
    """Helper print function."""

    print(f'dataset_name: {dataset_name}, model_name: {model_name}, embedding_name: {emb_name} -- {results_dct}\n')


def compute_centroids(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    emb_dim = X.shape[1]
    classes = set(y)
    num_classes = len(classes)

    centroids = np.zeros(shape=(num_classes, emb_dim))

    for label in range(num_classes):
        embeddings = X[y == label]
        num_embeddings = len(embeddings)

        for emb in embeddings:
            centroids[label, :] += emb

        centroids[label, :] /= num_embeddings

    return tf.convert_to_tensor(centroids, dtype=tf.float32)


def distance_metric(X, centroids, dist_type):
    X = np.asarray(X)
    centroids = np.asarray(centroids)

    num_embeddings = X.shape[0]
    num_centroids = centroids.shape[0]  # equivalent to num_classes

    if dist_type == 'euclidean':
        # modify arrays to shape (num_embeddings, num_centroids, emb_dim) in order to compare them
        x = np.repeat(X[:, np.newaxis, :], repeats=num_centroids, axis=1)
        centroids = np.repeat(centroids[np.newaxis, :, :], repeats=num_embeddings, axis=0)

        logits = tf.norm(x - centroids, ord='euclidean', axis=2)
    else:
        x_norm = tf.nn.l2_normalize(X, axis=1)
        centroids_norm = tf.nn.l2_normalize(centroids, axis=1)
        cos_sim = tf.matmul(x_norm, tf.transpose(centroids_norm))

        if dist_type == 'cosine':
            logits = 1 - cos_sim
        else:  # angular
            logits = tf.math.acos(cos_sim) / math.pi

    return tf.convert_to_tensor(logits)


def batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def get_intents_selection(lst, num_intents: int):
    """
    Returns a random selection of intent labels.
    :params:            lst - contains sublists in the following form: [sentence, label]
                        num_intents, int
    :returns:           selection, (num_intents, ) np.ndarray
    """

    unique_intents = list(set([l[1] for l in lst]))
    selection = np.random.choice(unique_intents, num_intents,
                                 replace=False)  # replace=False doesn't allow elements to repeat

    return selection


def filter(lst, selection, mode):
    ret = []

    for l in lst:
        if l[1] in selection:
            ret.append(l)
        else:
            if mode == 'test':
                ret.append([l[0], 'oos'])

    return ret


def get_f1_ood_id(y_test, y_pred, oos_label):
    """
    Compute separate F1 scores for the out-of-domain class and in-domain classes.
    Adapted from https://github.com/thuiar/Adaptive-Decision-Boundary/blob/9bcd4a8c6ccd3d50eaf04a89cb567f25f2a058f5/util.py#L64
    """

    conf_mat = confusion_matrix(y_test, y_pred)

    f1_ood = 0
    f1_id_lst = []
    num_classes = conf_mat.shape[0]

    for idx in range(num_classes):
        tp = conf_mat[idx][idx]  # true positives
        r = tp / conf_mat[idx].sum() if conf_mat[idx].sum() != 0 else 0  # recall
        p = tp / conf_mat[:, idx].sum() if conf_mat[:, idx].sum() != 0 else 0  # precision
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0  # f1

        if idx != oos_label:
            f1_id_lst.append(f)
        else:
            f1_ood = f * 100

    f1_id = np.mean(f1_id_lst) * 100

    return f1_ood, f1_id
