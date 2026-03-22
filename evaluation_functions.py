import time
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from abc import ABC, abstractmethod
from typing import Any, Optional

class ClassifierAbstraction(ABC):
    @abstractmethod
    def fit(self, X : np.ndarray, y : np.ndarray) -> None:
        pass

    @abstractmethod
    def decision_function(self, X : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

def measure_fit_time(classifier : ClassifierAbstraction | Any, X_train : np.ndarray, y_train : np.ndarray, n_runs=100):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        classifier.fit(X_train, y_train)
        times.append(time.perf_counter() - t0)
    return np.mean(times)

def print_metrics(classifier : ClassifierAbstraction | Any, X_test : np.ndarray, y_test : np.ndarray, name):
    pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print(f"\n── {name} ──")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(f"Accuracy:       {accuracy_score(y_test, pred)}")
    print(f"Sensitivity:    {tp / (tp + fn)}")
    print(f"Specificity:    {tn / (tn + fp)}")

def plot_roc(classifier : ClassifierAbstraction | Any, X_test : np.ndarray, y_test : np.ndarray, ax : Axes, title : Optional[str] = "Roc fig", color : Optional[str] = "blue"):
    scores = classifier.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax.plot(fpr, tpr, color=color)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}')

def plot_decision_boundary(classifier : ClassifierAbstraction | Any, X_test : np.ndarray, y_test, ax : Axes, title):
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', linewidths=0.5)
    ax.set_title(title)