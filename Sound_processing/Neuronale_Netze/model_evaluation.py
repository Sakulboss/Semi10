import training_data

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as pl

def model_evaluation(**kwargs):
    data = training_data(**kwargs)
    trained_model = kwargs.get('trained_model', None)
    X_test_norm = kwargs.get('X_test_norm', data[4])
    y_test = kwargs.get('y_test', data[6])
    unique_classes = kwargs.get('unique_classes', data[7])
    n_sub = kwargs.get('n_sub', data[8])
    printing = kwargs.get('printinge', False)

    if trained_model is None:
        model = model_training(**kwargs)[0]
    else: model = trained_model

    if printing: print("Shape of the test data: {}".format(X_test_norm.shape))
    y_test_pred = model.predict(X_test_norm)
    if printing: print("Shape of the predictions: {}".format(y_test_pred.shape))

    # The model outputs in each row 5 probability values (they always add to 1!) for each class.
    # We want to take the class with the highest probability as prediction!

    y_test_pred = np.argmax(y_test_pred, axis=1)
    if printing: print(y_test_pred)
    if printing: print("Shape of the predictions now: {}".format(y_test_pred.shape))

    # Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    if printing: print("Accuracy score = ", accuracy)

    if kwargs.get('confusion_matrix', False):
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

model_evaluation()


