from src.automllib.models.DumbDenseNet import DumbDenseNet
from src.automllib.models.BaseModelWrapper import BaseModelWrapper
from keras.metrics import Accuracy
from keras.datasets import imdb

import keras as K
import numpy as np
import logging


logger = logging.getLogger(__name__)


def test_ddn_model_compilation():
    """__init__ procedures"""
    params = {
        'metric_fn': Accuracy(),
        'epochs': 10,
        'batch_size': 32,
        'validation_split': 0.2,
        'input_shape': (10000,)
    }

    net = DumbDenseNet(params=params)
    assert type(net.get_model.get_layer(index=0)) == K.layers.core.dense.Dense


def test_ddn_imdb_train():
    """test model on binary classification dataset (minimal size)"""
    params = {
        'metric_fn': Accuracy(),
        'epochs': 3,
        'batch_size': 8,
        'validation_split': 0.2,
        'input_shape': (50,)
    }

    net = DumbDenseNet(params=params)

    # Data preprocessing
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=50)

    def vectorize_sequences(sequences, dimension=50):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    m, res = net.get_best_model(x_train, y_train, x_test, y_test)

    assert type(m) == K.models.Sequential
