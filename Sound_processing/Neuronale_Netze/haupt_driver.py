from datensatz import directory, dataset
from labeler import labeler
from training_data import training_data
from Sound_processing.Neuo_Netze_torch.model_processing_torch import model_training, model_evaluation
from mel_specs import mel_specs

def main(settings):
    directory()

    dir_list = dataset(big = False)
    labels = labeler(dir_list)
    mels = mel_specs(labels)
    trained_data = training_data(mels)
    trained = model_training(trained_data)
    model_evaluation(trained_data, trained_model=trained[0])



args = {
    'printing' : True
}

if __name__=='__main__':
    main(args)


