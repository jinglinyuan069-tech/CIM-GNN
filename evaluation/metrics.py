import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def regression_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae


def classification_metrics(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true)

    y_prob = np.asarray(y_prob).reshape(-1)

    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    return {
        "Accuracy": acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc
    }


def error_distribution(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_pred - y_true

    return error


def relative_improvement(baseline, model):

    baseline = np.asarray(baseline)
    model = np.asarray(model)

    improvement = (baseline - model) / baseline

    return improvement.mean()