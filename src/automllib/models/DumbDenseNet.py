from .BaseModelWrapper import BaseModelWrapper

from keras.layers import Dense
from keras import Sequential
from keras.callbacks import EarlyStopping


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
                      metrics=[self.metric])
        self.model = model
        self.logger.debug(self.summary())

    def summary(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def get_best_model(self, x_train, y_train, x_test, y_test):
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
        return self.model, self.model.evaluate(x_test, y_test)[1]