from utils import Split, batches
from custom_models import ADBPretrainCosFaceModel, ADBPretrainTripletLossModel

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers
import tensorflow_addons as tfa
import time


def create_embed_f(old_embed_f, dataset, limit_num_sents, type: str):
    """Fine-tunes embeddings from USE or SBERT (using their embedding function). Returns new embed function."""

    start_time = time.time()

    split = Split(old_embed_f)
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')

    emb_dim = X_train.shape[1]
    num_classes = len(set(np.asarray(y_train)))

    if type == 'cosface':
        model = ADBPretrainCosFaceModel(emb_dim, num_classes)
    else:  # triplet_loss
        model = ADBPretrainTripletLossModel(emb_dim)

    if type == 'cosface':
        loss = losses.SparseCategoricalCrossentropy()
        shuffle = True  # default
        batch_size = None  # defaults to 32
    else:  # triplet_loss
        loss = tfa.losses.TripletSemiHardLoss()
        shuffle = True  # shuffle before every epoch in order to guarantee diversity in pos and neg samples
        batch_size = 300  # same as above - to guarantee...

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=loss)

    if type == 'cosface':
        X = [X_train, y_train]
    else:  # triplet_loss
        X = X_train

    model.fit(X, y_train, epochs=20, shuffle=shuffle, batch_size=batch_size)

    def embed_f(X):
        embeddings_lst = []

        for batch in batches(X, 32):  # iterate in batches of size 32
            X = old_embed_f(batch)

            temp_emb = model(X)
            embeddings_lst.append(temp_emb)

        embeddings = tf.concat(embeddings_lst, axis=0)

        return embeddings

    end_time = time.time()
    time_pretraining = round(end_time - start_time, 1)

    return embed_f, time_pretraining
