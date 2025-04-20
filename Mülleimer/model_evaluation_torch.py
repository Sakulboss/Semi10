import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as pl


printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    if printing: print(*text, sep=sep, end=end, file=file)

def model_evaluation_torch(data, model, setting):
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', file_)
    X_test_norm = setting.get('X_test_norm', data[4])
    y_test = setting.get('y_test', data[6])
    unique_classes = setting.get('unique_classes', data[7])
    n_sub = setting.get('n_sub', data[8])
    printing = setting.get('printinge', False)


    printer("Shape of the test data: {}".format(X_test_norm.shape))

    # Convert the input data to a PyTorch tensor and ensure it's in the right shape
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)


    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculation for inference
        y_test_pred_logits = model(X_test_tensor)
        y_test_pred_probs = F.softmax(y_test_pred_logits, dim=1)  # Apply softmax to get probabilities

    printer("Shape of the predictions: {}".format(y_test_pred_probs.shape))
    # Get the predicted class by taking the argmax
    y_test_pred = torch.argmax(y_test_pred_probs, dim=1).numpy()
    printer(y_test_pred)
    printer("Shape of the predictions now: {}".format(y_test_pred.shape))

    # Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    printer("Accuracy score = ", accuracy)

    if setting.get('confusion_matrix', False):
        pl.figure(figsize=(10, 10))  # Adjusted size for better visibility
        cm = confusion_matrix(y_test, y_test_pred).astype(np.float32)
        # Normalize to probabilities
        for i in range(cm.shape[0]):
            if np.sum(cm[i, :]) > 0:
                cm[i, :] /= np.sum(cm[i, :])  # Convert counts to probabilities
        pl.imshow(cm, interpolation='nearest')
        ticks = np.arange(n_sub)
        pl.xticks(ticks, unique_classes)
        pl.yticks(ticks, unique_classes)
        pl.colorbar()
        pl.title('Confusion Matrix')
        pl.xlabel('Predicted Label')
        pl.ylabel('True Label')
        pl.show()
    return unique_classes


if __name__ == '__main__':
    pass