import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score

def confusion_matrix(y_true, y_pred):
    """
    Retorna a matriz de confusão.
    """
    labels = sorted(np.unique(y_true))
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum(np.logical_and(y_pred == pred_label, y_true == true_label))
    
    return cm

def confusion_matrix_plot(y_test, y_pred, ax=None):
    """
    Plota a matriz de confusão.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(y_test)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(cm, annot=True, linewidths=0.5, square=True, cmap='crest', xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_title(f'Accuracy Score: {np.sum(np.diag(cm)) / np.sum(cm):.2f}', size=15)
    ax.set_ylabel('True label', size=12)
    ax.set_xlabel('Predicted label', size=12)
    
    return ax

class Metrics:

    def __init__(self, digits):
        self.digits=sorted(digits)

    def accuracy(self, y_true, y_pred):
        """
        Retorna a acurácia.
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    def error(self, y_true, y_pred):
        """
        Retorna o erro.
        """
        return 1 - accuracy_score(y_true, y_pred)
    
    def precision(self, y_true, y_pred, label):
        """
        Retorna a precisão.
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] \
            / np.sum(cm[:, self.digits.index(label)])
    
    def recall(self, y_true, y_pred, label):
        """
        Retorna o recall.
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] /\
            np.sum(cm[self.digits.index(label), :])
    
    def f1_score(self, y_true, y_pred, label):
        """
        Retorna o F1 Score.
        """
        return 2 * self.precision_all(y_true, y_pred, label) * self.recall_all(y_true, y_pred, label)\
            / (self.precision_all(y_true, y_pred, label) + self.recall_all(y_true, y_pred, label))
    
    def weighted_f1_score(self, y_true, y_pred):
        """
        Retorna o F1 Score ponderado.
        """
        return sum([self.f1_score_all(y_true, y_pred, label) for label in self.digits]) / len(self.digits)
    
    def plot_cm(self, y_test, y_pred, ax=None):
        """
        Plota a matriz de confusão.
        """
        confusion_matrix_plot(y_test, y_pred, ax=ax)
        plt.show()

    def print_metrics(self, y_true, y_pred):
        """
        Imprime as métricas.
        """
        print(f'Acurácia: {self.accuracy(y_true, y_pred):.2f}')
        print(f'Erro: {self.error(y_true, y_pred):.2f}')
        for label in self.digits:
            print(f'Precisão ({label}): {self.precision(y_true, y_pred, label):.2f}')
            print(f'Recall ({label}): {self.recall(y_true, y_pred, label):.2f}')
            print(f'F1 Score ({label}): {self.f1_score(y_true, y_pred, label):.2f}')
        print(f'F1 Score Ponderado: {self.weighted_f1_score(y_true, y_pred):.2f}')
    