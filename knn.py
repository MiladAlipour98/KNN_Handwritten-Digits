from collections import Counter

import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    a = a.flatten()
    b = b.flatten()
    return np.sqrt(np.sum((a - b) ** 2))


def nearest_neighbors(train_set, x_sample, k=1):
    xs, ys = train_set
    distances = np.array([distance(x_sample, x) for x in xs])
    mask = distances.argsort()[:k]
    return ys[mask]


def predict_sample(train_set, x_sample, k=1):
    neighbors = nearest_neighbors(train_set, x_sample, k)
    return Counter(neighbors).most_common(1)[0][0]


def predict(train_set, xs_sample, k=1):
    predictions = np.array([predict_sample(train_set, x_sample, k) for x_sample in xs_sample])
    return predictions


def calculate_error(ys: np.ndarray, predictions: np.ndarray):
    return np.sum(ys != predictions) / len(ys)