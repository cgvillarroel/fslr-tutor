import csv
import math
import matplotlib.pyplot as plt
# import numpy as np # in case of metrics divisor of 0
from collections.abc import Callable
from sklearn import metrics
from .types import Result


def cosine(vec1, vec2):
    vec_len = len(vec1)
    sum_of_prod = 0
    sum_of_square_1 = 0
    sum_of_square_2 = 0
    for i in range(vec_len):
        sum_of_prod += vec1[i] * vec2[i]
        sum_of_square_1 += vec1[i] * vec1[i]
        sum_of_square_2 += vec2[i] * vec2[i]

    return sum_of_prod / (math.sqrt(sum_of_square_1) * (math.sqrt(sum_of_square_2)))


def euclideanDistance(point1, point2):
    x = point2.x - point1.x
    y = point2.y - point1.y
    return math.sqrt((x ** 2) + (y ** 2))


def accuracy(mat):
    return (mat[0] + mat[1]) / sum(mat)


def precision(mat):
    all_positive = mat[0] + mat[2]
    return mat[0] / all_positive if all_positive > 0 else 0


def recall(mat):
    all_true = mat[0] + mat[1]
    return mat[0] / all_true if all_true > 0 else 0


def f1Score(mat):
    denominator = (2 * mat[0] + mat[2] + mat[3])
    return (2 * mat[0]) / denominator if denominator > 0 else 0


def printStats(mat):
    print(f"TP: {mat[0]}\t FN: {mat[3]}")
    print(f"FP: {mat[2]}\t TN: {mat[1]}\n")

    print(f"Accuracy : {accuracy(mat)}")
    print(f"Precision: {precision(mat)}")
    print(f"Recall   : {recall(mat)}")
    print(f"F1 Score : {f1Score(mat)}")


def importGestureDifferences(file_name: str, count: int):
    lut = [[{}] * count for _ in range(count)]

    for i in range(count):
        lut[i][i] = {
            "location": True,
            "motion": True,
            "shape": True,
            "face": True,
        }

    with open(file_name, newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            gesture1 = int(row[0])
            gesture2 = int(row[1])
            lut[gesture1][gesture2] = {
                "location": row[2] == "Same",
                "motion": row[3] == "Same",
                "shape": row[3] == "Same",
                "face": row[4] == "Same",
            }
            lut[gesture2][gesture1] = {
                "location": row[2] == "Same",
                "motion": row[3] == "Same",
                "shape": row[3] == "Same",
                "face": row[4] == "Same",
            }

    return lut


def train_test_split(x: list, y: list, train_size: float):
    split_point = int(len(x) * train_size)
    x_train = x[:split_point]
    x_test = x[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]

    return x_train, x_test, y_train, y_test


def train_test_folds(x: list, y: list, fold_count: int):
    x_train_folds = []
    x_test_folds = []
    y_train_folds = []
    y_test_folds = []

    interval = len(x) / fold_count
    for i in range(fold_count):
        split_point1 = int(i * interval)
        split_point2 = int((i + 1) * interval)
        x_train_folds.append(x[:split_point1] + x[split_point2:])
        x_test_folds.append(x[split_point1: split_point2])
        y_train_folds.append(y[:split_point1] + y[split_point2:])
        y_test_folds.append(y[split_point1: split_point2])

    return x_train_folds, x_test_folds, y_train_folds, y_test_folds

# %% [md]
# ## Modules

# %%


def test_location(result: Result, thresholds: dict[str, float]):
    return (
        result.location_results[0] >= thresholds["location"]
        and result.location_results[1] >= thresholds["location"]
    )

# %%


def test_motion(result: Result, thresholds: dict[str, float]):
    return (
        result.motion_results[0] <= thresholds["motion_shoulder"]
        and result.motion_results[1] <= thresholds["motion_shoulder"]
        and result.motion_results[2] <= thresholds["motion_elbow"]
        and result.motion_results[3] <= thresholds["motion_elbow"]
        and result.motion_results[4] <= thresholds["motion_wrist"]
        and result.motion_results[5] <= thresholds["motion_wrist"]
    )

# %%


def test_shape(result: Result, thresholds: dict[str, float]):
    return (
        result.shape_results[0] >= thresholds["shape"]
        and result.shape_results[1] >= thresholds["shape"]
    )

# %%


def test_face(result: Result, thresholds: dict[str, float]):
    return result.face_result >= thresholds["face"]

# %% [md]
# ## Combined Modules
# %%


def test_with_face_binary(result: Result, thresholds: dict[str, float]):
    return (
        test_location(result, thresholds)
        and test_motion(result, thresholds)
        and test_shape(result, thresholds)
        and test_face(result, thresholds)
    )

# %% [md]
# ## Finding thresholds
# %%


def plot_thresholds(
        threshold_name: str,
        iterator: range,
        test_function: Callable,
        x_values: list[Result],
        y_values: list[Result],
        other_thresholds: dict[str, float] = {},
        scale=1000):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    thresholds = [i / scale for i in iterator]
    threshold_dicts = [{threshold_name: i / 1000} for i in iterator]
    _ = [threshold.update(other_thresholds) for threshold in threshold_dicts]

    for threshold in threshold_dicts:
        y_pred = [test_function(x, threshold) for x in x_values]
        accuracy_scores.append(metrics.accuracy_score(
            y_pred,
            y_values))
        precision_scores.append(metrics.precision_score(
            y_pred,
            y_values,
            zero_division=0.0))
        recall_scores.append(metrics.recall_score(
            y_pred,
            y_values,
            zero_division=0.0))
        f1_scores.append(metrics.f1_score(
            y_pred,
            y_values,
            zero_division=0.0))

    plt.plot(thresholds, accuracy_scores, color="r", label="accuracy")
    plt.plot(thresholds, precision_scores, color="g", label="precision")
    plt.plot(thresholds, recall_scores, color="b", label="recall")
    plt.plot(thresholds, f1_scores, color="y", label="f1 score")


def test_thresholds(
        x_values: list[float],
        y_values: list[float],
        thresholds: dict[str, float],
        test_function: Callable):
    y_pred = [test_function(x, thresholds) for x in x_values]
    confusion_matrix = metrics.confusion_matrix(y_values, y_pred)

    cm_plot = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=[
            "Incorrect/Different Gesture",
            "Correct/Same Gesture",
        ])
    cm_plot.plot()

    print(f"Accuracy : {metrics.accuracy_score(y_values, y_pred)}")
    print(f"Precision: {metrics.precision_score(y_values, y_pred, zero_division=0.0)}")
    print(f"Recall   : {metrics.recall_score(y_values, y_pred, zero_division=0.0)}")
    print(f"F1 Score : {metrics.f1_score(y_values, y_pred, zero_division=0.0)}")
