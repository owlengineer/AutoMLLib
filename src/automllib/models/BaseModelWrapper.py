import logging
import importlib

from abc import abstractmethod


class BaseModelWrapper:
    """Wrapper class for ML-models"""

    def __init__(self,
                 params: dict):
        self.config = params
        self.model = None
        self._init_metric()
        self._init_logger()
        self._init_model()

    @abstractmethod
    def _init_model(self):
        """saves model Python-object into self.model"""
        pass

    def _init_metric(self):
        """init metric function"""
        self.metric = self.config['metric_fn']

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