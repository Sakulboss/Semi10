import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(y_test, y_test_pred, n_sub, unique_classes):
    pl.figure(figsize=(50, 50))
    cm = confusion_matrix(y_test, y_test_pred).astype(np.float32)
    # normalize to probabilities
    for i in range(cm.shape[0]):
        if np.sum(cm[i, :]) > 0:
            cm[i, :] /= np.sum(cm[i, :])  # by dividing through the sum, we convert counts to probabilities
    pl.imshow(cm)
    ticks = np.arange(n_sub)
    pl.xticks(ticks, unique_classes)
    pl.yticks(ticks, unique_classes)
    pl.show()