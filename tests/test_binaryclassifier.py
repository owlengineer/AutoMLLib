from src.automllib.BinaryClassifier import BinaryClassifier
from keras.metrics import Accuracy
from keras.datasets import imdb

import keras as K
import numpy as np
import logging


logger = logging.getLogger(__name__)


def test_binary_classifier():
    params = {
        'metric_fn': 'accuracy',
        'epochs': 10,
        'batch_size': 32,
        'validation_split': 0.2,
        'input_shape': (10000,)
    }

    # Data preprocessing
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    logger.info(f"X_tr: {x_train.shape}, Y_tr: {y_train.shape}")
    classifier = BinaryClassifier(params=params)
    m_obj = classifier.fit_best(x_train, y_train, x_test, y_test, attempts=2, random_factor=2)

    logger.info(f"Best model test metric calc (loss, metric): {m_obj.evaluate(x_test, y_test)}")

    assert type(m_obj.get_layer(index=0)) == K.layers.core.dense.Dense