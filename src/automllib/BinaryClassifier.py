import copy

from .classifiers.DumbDenseNet import DumbDenseNet
import logging


class BinaryClassifier:
    """Wrapper class, that finds the best implemented binary classification model for current task"""

    def __init__(self,
                 params: dict = []):
        self.params = params
        # ToDo это хорошо бы переписать так, чтобы модели подгружались динамически
        #  когда будут добавляться новые модели, не нужно будет здесь ничего менять
        self.models = [
            DumbDenseNet(params=params)
        ]
        self.best_model = None
        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)

    def fit_best(self, x_train, y_train, x_test, y_test, attempts: int = 1, random_factor: int = 1):
        """fit N times (un)randomized models"""
        best_result = None
        best_model = None
        for m in self.models:
            self.logger.info(f"{type(m)} model training...")
            for i in range(attempts):
                m.randomize_params(factor=random_factor)
                m_obj = m.train(x_train, y_train)
                metric_res = m.predict(x_test, y_test)
                self.logger.info(f"Attempt {i}: {metric_res}, best: {best_result}")
                if not best_result or metric_res > best_result:
                    best_result = metric_res
                    best_model = copy.deepcopy(m_obj)
        return best_model
