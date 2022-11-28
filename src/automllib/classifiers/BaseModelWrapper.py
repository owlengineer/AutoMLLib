import logging
import importlib

from abc import abstractmethod


class BaseModelWrapper:
    """Wrapper class for ML-models"""

    def __init__(self,
                 params: dict):
        self.model = None
        self.config = None
        self._init_default_config()
        self._init_config(params)
        self._init_logger()
        self._init_model()

    @abstractmethod
    def _init_model(self):
        """saves model Python-object into self.model"""
        pass

    @abstractmethod
    def _init_default_config(self):
        """returns default configurations dict() """
        pass

    @abstractmethod
    def _init_config(self, params):
        """init model config"""
        pass

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)

    # ToDo в этот proof of concept можно добавить больше интерактива -- например, словарь параметр-коэффициент
    @abstractmethod
    def _randomize_params(self, factor: int = 1):
        """randomize hyperparams of model by factor. How factor forces on params -- implemented in models classes"""
        pass

    @property
    def get_model(self):
        return self.model

    @abstractmethod
    def summary(self):
        """text summary of model"""
        pass

    @abstractmethod
    def train(self, x_train, y_train):
        """returns best state of current model by training on given data"""
        pass

    @abstractmethod
    def predict(self, x_test, y_test):
        """returns prediction for given data"""
        pass