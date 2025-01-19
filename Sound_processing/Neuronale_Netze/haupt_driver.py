import datensatz as ds
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from datensatz import directory, dataset
from labeler import labeler

def main(settings):
    directory()
    dir_list = dataset(settings)
    labeler(dir_list, settings)


args = {
    'printing' : True
}

if __name__=='__main__':
    main(args)


