from custom_layers import CosFace

import tensorflow as tf
from tensorflow.keras import layers, activations


class ADBPretrainCosFaceModel(tf.keras.Model):
    """Adaptive Decision Boundary with CosFace pre-training model using USE or SBERT embeddings."""

    def __init__(self, emb_dim, num_classes):
        super(ADBPretrainCosFaceModel, self).__init__()
        self.inp = layers.Input(shape=(emb_dim))
        self.labels = layers.Input(shape=(1))

        self.dense = layers.Dense(emb_dim, activation=activations.relu)
        self.dropout = layers.Dropout(0.1)
        self.dense2 = layers.Dense(emb_dim, activation=activations.relu)
        self.dense3 = layers.Dense(emb_dim, activation=activations.relu)
        self.cosface = CosFace(num_classes=num_classes)

    def call(self, inputs, training=None):
        if training:
            inputs, labels = inputs

        x = self.dense(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)

        if training:
            x = self.dropout(x)
            probs = self.cosface([x, labels], training)

            return probs

        return tf.nn.l2_normalize(x, axis=1)  # return normalized embeddings


class ADBPretrainTripletLossModel(tf.keras.Model):
    """Adaptive Decision Boundary with Triplet Loss pre-training model using USE or SBERT embeddings."""

    def __init__(self, emb_dim):
        super(ADBPretrainTripletLossModel, self).__init__()
        self.inp = layers.Input(shape=(emb_dim))
        self.dense = layers.Dense(emb_dim, activation=activations.relu)
        self.dropout = layers.Dropout(0.1)
        self.dense2 = layers.Dense(emb_dim, activation=activations.relu)
        self.dense3 = layers.Dense(emb_dim, activation=activations.relu)
        self.dense4 = layers.Dense(emb_dim, activation=None)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)

        if training:
            x = self.dropout(x)
            x = self.dense4(x)

        return tf.nn.l2_normalize(x, axis=1)  # return normalized embeddings
