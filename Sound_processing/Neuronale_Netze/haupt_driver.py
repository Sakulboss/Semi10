import datensatz as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from datensatz import directory, dataset
from labeler import labeler
from training_data import training_data

def main(settings):
    directory()
    dir_list = dataset(settings)
    labels = labeler(dir_list, settings)
    trained_data = training_data(labels, settings)


args = {
    'printing' : True
}

if __name__=='__main__':
    main(args)


