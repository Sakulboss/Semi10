from Sound_processing.Neuo_Netze_torch.model_processing_torch import *
import sys
import numpy as np

printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    if printing: print(*text, sep=sep, end=end, file=file)

def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    counter = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            counter += 1
    return round(counter / len(y_true), 3)

def get_class(y_pred):
    return_list = []
    for i in y_pred:
        max_value, pos = 0, 0
        for k in range(len(i)):
            print(i[k], max_value)
            if i[k] > max_value:
                max_value = i[k]
                pos = k

        return_list.append(k)
    return return_list

def model_evaluation(data, model, setting):
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)
    X_test_norm = setting.get('X_test_norm', data[4])
    y_test = setting.get('y_test', data[6])
    unique_classes = setting.get('unique_classes', data[7])
    n_sub = setting.get('n_sub', data[8])
    printing = setting.get('printing', False)


    printer("Shape of the test data: {}".format(X_test_norm.shape))
    y_test_pred = model.predict(X_test_norm)
    printer("Shape of the predictions: {}".format(y_test_pred.shape))
    printer(y_test_pred)

    y_test_pred_new = np.argmax(y_test_pred, axis=1)
    printer(y_test_pred_new)
    printer(get_class(y_test_pred))
    printer("Shape of the predictions: {}".format(y_test_pred_new.shape))

    # Accuracy
    acc = accuracy(y_test_pred_new, y_test)
    printer("Accuracy score = ", acc)

    return acc, unique_classes




if __name__ == '__main__':
    pass


