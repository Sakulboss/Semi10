from Sound_processing.training_files.driver_mels import trainingdata
from model_processing_tf import model_training
from model_evaluation_tf import model_evaluation


def main(settings):

    trained_data: tuple = trainingdata(settings)
    trained: tuple      = model_training(trained_data, settings)
    model_evaluation(trained_data, trained[0], settings)



args = {
    'printing' : True,
    'size'     : 'small',
    'model'    : 'torch',
}

if __name__=='__main__':
    main(args)


