#!/usr/bin/env python3
import numpy as np

# Функция активации (сигмоида)
# Необходима для определения значения весов
def sigmoid(x, der=False):
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Набор входных данных
x = np.array([[1, 0, 1],
              [1, 0, 1],
              [0, 1, 0],
              [0, 1, 0]])

# Выходные данные
y = np.array([[0, 0, 1, 1]]).T # T - функция переноса

# Сделаем случайные числа более определенными
np.random.seed(1)

# Инициализируем веса случайным образом со средним 0
syn0 = 2 * np.random.random((3, 1)) - 1

k1 = []

for iter in range(10000):
    # Прямое распространение
    k0 = x
    k1 = sigmoid(np.dot(k0, syn0))

    # Насколько мы ошиблись?
    k1_error = y - k1

    # Перемножим это с наклоном сигмоиды
    # на основе значений в k1
    k1_delta = k1_error * sigmoid(k1, True)

    # Обновим веса
    syn0 += np.dot(k0.T, k1_delta)

print("Выходные данные после тренеровки:")
print(k1)


new_one = np.array([0, 1, 0])
k1_new = sigmoid(np.dot(new_one, syn0))
print("Новые данные:")
print(k1_new)
























