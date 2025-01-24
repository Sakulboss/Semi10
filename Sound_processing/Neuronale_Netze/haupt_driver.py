from datensatz import directory, dataset
from labeler import labeler
from training_data import training_data
from model_processing import model_training

def main(settings):
    directory()
    dir_list = dataset(big = False)
    labels = labeler(dir_list)
    trained_data = training_data(labels)
    model_training(trained_data)


args = {
    'printing' : True
}

if __name__=='__main__':
    main(args)


