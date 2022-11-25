import logging
import importlib

from abc import abstractmethod
from ..utils import merge_dicts
from keras.metrics import Accuracy


class BaseModelWrapper:
    """Wrapper class for ML-models"""

    def __init__(self,
                 params: dict):
        self.model = None
        self._init_config(params)
        self._init_logger()
        self._init_model()

    @abstractmethod
    def _init_model(self):
        """saves model Python-object into self.model"""
        pass

    def _init_config(self, params):
        """init model config"""
        self.config = {
            'metric_fn': Accuracy(),
            'epochs': 20,
            'batch_size': 64,
            'validation_split': 0.2,
            'input_shape': (100,)
        }
        merge_dicts(self.config, params)

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)

    @property
    def get_model(self):
        return self.model

    @abstractmethod
    def summary(self):
        """text summary of model"""
        pass

    @abstractmethod
    def get_best_model(self, x_train, y_train, x_test, y_test):
        """returns best state of current model by training on given data"""
        pass