from Sound_processing.training_files.driver_mels import trainingdata
from Mülleimer.model_processing_torch import model_training_torch
#from model_evaluation_torch import model_evaluation
import sys

file_ = sys.stdout
printing = True

def main(settings):

    trained_data: tuple = trainingdata(settings)
    trained: tuple      = model_training_torch(trained_data, settings)
    #model_evaluation(trained_data, trained[0], settings)
    print(4)
    # model_training_torch(trained_data, settings)
    # model_evaluation(trained_data, trained[0], settings)


args = {
    'printing' : True,
    'size'     : 'small',
    'model'    : 'torch',
}

if __name__=='__main__':
    main(args)




