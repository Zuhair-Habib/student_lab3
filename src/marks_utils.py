
import numpy as np


def standardise_loop(data):
    data = np.asarray(data, dtype=float)
    rows, cols = data.shape

    means = []
    stds = []

    # Calculate mean and standard deviation of each column
    for j in range(cols):
        total = 0.0

        for i in range(rows):
            total += data[i, j]

        mean = total / rows
        means.append(mean)

        squared_total = 0.0

        for i in range(rows):
            squared_total += (data[i, j] - mean) ** 2

        std = (squared_total / rows) ** 0.5
        stds.append(std)

    # Standardise every element
    result = np.zeros((rows, cols), dtype=float)

    for i in range(rows):
        for j in range(cols):
            if stds[j] == 0:
                result[i, j] = 0.0
            else:
                result[i, j] = (
                    (data[i, j] - means[j]) / stds[j]
                )

    return result


def standardise_np(data):
    data = np.asarray(data, dtype=float)

    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)

    safe_stds = np.where(stds == 0, 1.0, stds)

    return (data - means) / safe_stds
