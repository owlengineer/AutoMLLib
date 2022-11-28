from .BaseModelWrapper import BaseModelWrapper

from keras.layers import Dense
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.metrics import Accuracy
from ..utils import merge_dicts

import random


class DumbDenseNet(BaseModelWrapper):
    """Dense layered net for bin classification"""

    def __init__(self,
                 params):
        super().__init__(params=params)

    def _init_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=self.config['input_shape']),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[self.config['metric_fn']])
        self.model = model
        self.logger.debug(self.summary())

    def _init_default_config(self):
        self.default_config = {
            'metric_fn': Accuracy(name="accuracy"),
            'epochs': 20,
            'batch_size': 64,
            'validation_split': 0.2,
            'input_shape': (100,)
        }

    def _init_config(self, params):
        merge_dicts(self.default_config, params)
        self.config = self.default_config

    def randomize_params(self, factor: int = 1):
        self.config['epochs'] = random.randrange(int(self.default_config['epochs']/factor),
                                                 self.default_config['epochs']*2)
        self.config['batch_size'] = random.choice([int(self.default_config['batch_size']/factor),
                                                   self.default_config['batch_size']*2])

    def summary(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def train(self, x_train, y_train):
        """returns best weighted model and metric eval, test/train data need to be prepared (vectorising)"""
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',  # TODO прописать мониторинг по test выборке вместо val
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True  # key argument, tricky saving best weights
        )
        H = self.model.fit(x_train,
                       y_train,
                       epochs=self.config['epochs'],
                       batch_size=self.config['batch_size'],
                       validation_split=self.config['validation_split'],
                       callbacks=[early_stopping_monitor],
                       verbose=0)
        self.config = self.default_config
        return self.model

    def predict(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]