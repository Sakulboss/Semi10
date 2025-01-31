from Sound_processing.Neuronale_Netze.model_processing import *
import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as pl
import torch

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

    # Ensure the model is in evaluation mode
    model.eval()

    # Convert the test data to a PyTorch tensor and move to the appropriate device
    X_test_norm_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    if torch.cuda.is_available():
        X_test_norm_tensor = X_test_norm_tensor.cuda()
        model = model.cuda()

    printer("Shape of the test data: {}".format(X_test_norm_tensor.shape))

    # Disable gradient calculation for inference
    with torch.no_grad():
        y_test_pred_logits = model(X_test_norm_tensor)

    # Get the predicted class by taking the argmax
    y_test_pred = torch.argmax(y_test_pred_logits, dim=1).cpu().numpy()
    printer("Shape of the predictions: {}".format(y_test_pred.shape))

    # Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    printer("Accuracy score = {}".format(accuracy))

    if setting.get('confusion_matrix', False):
        pl.figure(figsize=(10, 10))  # Adjust the figure size as needed
        cm = confusion_matrix(y_test, y_test_pred).astype(np.float32)
        # Normalize to probabilities
        for i in range(cm.shape[0]):
            if np.sum(cm[i, :]) > 0:
                cm[i, :] /= np.sum(cm[i, :])  # Convert counts to probabilities
        pl.imshow(cm, interpolation='nearest', cmap=pl.cm.Blues)
        ticks = np.arange(n_sub)
        pl.xticks(ticks, unique_classes, rotation=45)
        pl.yticks(ticks, unique_classes)
        pl.colorbar()
        pl.xlabel('Predicted label')
        pl.ylabel('True label')
        pl.title('Confusion Matrix')
        pl.show()


if __name__ == '__main__':
    pass


