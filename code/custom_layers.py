import tensorflow as tf
from tensorflow.keras import layers


class CosFace(layers.Layer):
    """
    Implementation of CosFace layer. Reference: https://arxiv.org/abs/1801.09414
    Credits: https://www.kaggle.com/chankhavu/keras-layers-arcface-cosface-adacos

    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """

    def __init__(self,
                 num_classes,
                 s=64.0,
                 m=0.35,
                 regularizer=None,
                 name='cosface',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)

    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """

        if training:
            embeddings, labels = inputs
            # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
            labels = tf.reshape(labels, [-1], name='label_shape_correction')
        else:
            embeddings = inputs

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embeddings, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        if training:
            one_hot_labels = tf.one_hot(labels,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            final_theta = tf.where(tf.cast(one_hot_labels, dtype=tf.bool),
                                   cosine_sim - self._m,
                                   cosine_sim,
                                   name='cosine_sim_with_margin')
            output = self._s * final_theta
        else:
            # We don't have labels if we're not in training mode
            output = self._s * cosine_sim

        return tf.nn.softmax(output)
