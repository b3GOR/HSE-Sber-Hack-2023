# HSE-Sber_HACK-2023

Задача: на основе данных о транзакциях клиента, предсказать его пол

## Бэкенд-разработчики:
- [Михаил](https://github.com/varekha)
- [Слава](ttps://github.com/02lava)
- [Егор](https://github.com/b3GOR)
- [Олег](https://github.com/Aureilius)

## Cтэк технологий

- scikit-learn
- seaborn
- XGBoost
- CatBoostClassifier
- Numpy



## Подготовка и запуск проекта уже обученной модели

### Склонируйте репозиторий:
```sh
https://github.com/b3GOR/HSE-Sber-Hack-2023.git
```

### Сохраните папку data

### Загрузите необходимые библиотеки
```sh
pip install -r requirements.txt
```
### Запустите inference.py
После запуска появится файл result.csv с id клиента и вероятностью пренадлежности к классу 1
```sh
result.csv
```

## Если необходимо обучить модель

### Зайти в файл train_catboost.ipynb 
Гиперпараметры для обучения
```sh
param_dist = {
    'iterations': 300,
    'learning_rate': 0.1,
    'depth': 3,
    'l2_leaf_reg': 3,
    'border_count': 200
}
```

### После обучения, необходимо сохранить модель
```sh
pred_model.save_model('имя_вашей_модели', format='cbm')
```

### Далее, записываем название и путь вашей модели в inference.py

```sh
MODEL_PATH = "путь_к_вашей_модели"


model = CatBoostClassifier()  
model.load_model(os.path.join(MODEL_PATH,'название_вашей_модели'))
```
После запуска появится файл result.csv с id клиента и вероятностью пренадлежности к классу 1
```sh
result.csv
```


