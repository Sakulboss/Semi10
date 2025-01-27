from Sound_processing.Neuronale_Netze.model_processing import *
import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as pl


printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    if printing: print(*text, sep=sep, end=end, file=file)

def model_evaluation(data, model, setting):
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', file_)
    X_test_norm = setting.get('X_test_norm', data[4])
    y_test = setting.get('y_test', data[6])
    unique_classes = setting.get('unique_classes', data[7])
    n_sub = setting.get('n_sub', data[8])
    printing = setting.get('printinge', False)


    printer("Shape of the test data: {}".format(X_test_norm.shape))
    y_test_pred = model.predict(X_test_norm)
    printer("Shape of the predictions: {}".format(y_test_pred.shape))

    # The model outputs in each row 5 probability values (they always add to 1!) for each class.
    # We want to take the class with the highest probability as prediction!

    y_test_pred = np.argmax(y_test_pred, axis=1)
    printer(y_test_pred)
    printer("Shape of the predictions now: {}".format(y_test_pred.shape))

    # Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    printer("Accuracy score = ", accuracy)

    if setting.get('confusion_matrix', False):
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


    return unique_classes

if __name__ == '__main__':
    pass


