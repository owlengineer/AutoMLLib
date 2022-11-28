# AutoMLLib

Данная библиотека предоставляет интерфейс для обучения моделей для задачи бинарной классификации.

# Features

1. Получение лучшей модели для поданных данных (на основе подсчета метрики каждую эпоху)
2. Рандомизация параметров отдельных моделей -- библиотека обучает каждую модель несколько раз, рандомизируя её параметры при каждом пайплайне. Так увеличивается шанс на получение более точной модели.

### ToDo 

1. Для увеличения точности -- обучение всего перечня доступных моделей (models package)
   1. Чтобы отсеивать модели с неподходящими параметрами -- создать checker входных параметров (через вот это https://python-jsonschema.readthedocs.io/en/stable/)
   2. Чтобы аналитик понял что было проверено -- по завершении обучения выводить summary в виде:
      1. Успех/неуспех(неподходящие параметры) - модель - результат метрики 
2. Для увеличения точности в нейросетевых решениях -- прописать package с блоками слоев (Inpu, Dense, DropOut и т.п), чтобы структуру нейросети тоже рандомизировать перемешивая блоки между собой

## Install (Linux) 

Как подключить библиотеку в ваш python-проект:

#### Requires

* Poetry PM (1.2.2+)
* Python 3.10/3.11

#### Установка в venv:
```commandline
(venv) $ poetry add git+ssh://git@github.com:owlengineer/AutoMLLib.git
```

## Usage

Пример обучения классификатора:
```python
from automllib.BinaryClassifier import BinaryClassifier

params = {
    'metric_fn': 'accuracy',
    'epochs': 10,
    'batch_size': 32,
    'validation_split': 0.2,
    'input_shape': (10000,)
}

classifier = BinaryClassifier(params=params)
model_object = classifier.fit_best(x_train, y_train, x_test, y_test, attempts=2, random_factor=2) 
```

Пример обучения конкретной модели классификатора:
```python
from automllib.classifiers.DumbDenseNet import DumbDenseNet
from keras.metrics import TruePositives

#
# ... prepare x_train, y_train, x_test, y_test data

# параметры, у каждой модели своя JSON-SCHEMA параметров 
params = {
        'metric_fn': TruePositives(),  # функцию можно реализовать самостроятельно или выбрать готовую
        'epochs': 10,
        'batch_size': 32,
        'validation_split': 0.2,
        'input_shape': (1000,)
    }
model = DumbDenseNet(params=params)
best_model = model.train(x_train, y_train)

```