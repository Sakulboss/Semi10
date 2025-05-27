from Sound_processing import mel_spec_calculator as msc
import numpy as np
from sklearn.preprocessing import StandardScaler
from Sound_processing.Neuro_Netze_torch.network_prep import CNN
from torch import load, tensor, cuda
import os, torch, logging


def setup_logging(args: dict) -> logging.Logger:
    handlers = []
    if args.get('log_to_file', False):   logging.FileHandler(args.get('log_file', 'training.log'))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=args.get('level', 10),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)



def predict(filepath: str, model_path: str, class_labels: list, logger, model_struct=None, channel=0):
    """
    This function loads the model and predicts the class of the audio file.
    Args:
        filepath:      str            of path to the audio file
        model_path:    str            of path to the model (needs to be saved with structure and weights)
        class_labels:  list           with the names of the classes
        logger:        loggingLogger  for logging
        model_struct:  str            with model architecture - if only weights are saved, this is needed to load the model
        channel:       int            which channel to use (0 for left or 1 for right)
    Returns:
        class name with its probability
    """
    #os.chdir('modelle')
    devi = torch.device("cuda" if cuda.is_available() else "cpu")
    if model_struct is None:
        model = load(model_path, weights_only=False)
    else:
        model = CNN(model_struct)
        model.load_state_dict(load(model_path, weights_only=True))
    model.eval()
    model.to(devi)
    logger.info("Model loaded")

    mel_spec = msc.compute_mel_spec_for_audio_file(filepath, mono=False, channel = channel)
    logger.info('Mel specs erstellt')
    count = 0
    results = []
    for mel in chunks(mel_spec, 100):
        if count == 0: logger.debug(f"Mel spectrogram: ", mel.shape); count = 1
        mel = np.expand_dims(mel, axis=0)  # channel dimension
        mel = np.expand_dims(mel, axis=0)  # batch dimension
        scaler = StandardScaler()
        original_shape = mel[0].shape
        mel_2d = mel[0].reshape(-1, original_shape[-1])
        mel_scaled = scaler.fit_transform(mel_2d)
        mel[0] = mel_scaled.reshape(original_shape)
        mel = tensor(mel).to(devi)

        results.append(predict_result(mel, model))
    res = calculate_highest(results)
    #index = calculate_highest(results)[0]
    #return class_labels[index], index
    return res

def predict_result(mel_spec, model):
    _, predictions = model(mel_spec).max(1)
    predictions_new = np.array(predictions.cpu().numpy(), copy=True)
    return predictions_new


def chunks(l, n):
    for i in range(0, len(l) - 1, n): assert len(l[i]) == len(l[i + 1])
    for i in range(0, len(l[0]), n):
        parts = []
        for k in l:
            if i + n < len(k):
                parts.append(k[i:i + n])
            else:
                parts.append(k[-n:])
        yield np.array(parts)


def calculate_highest(results):
    results = [int(i[0]) for i in results]
    results.sort()
    sublists = {}
    if len(results) == 2: return results[0]
    for item in results:
        if item not in sublists:
            sublists[item] = []
        sublists[item].append(item)
    new_results = list(sublists.values())
    new_results.sort(key=len, reverse=True)
    return_results = []
    for i in range(len(new_results)):
        return_results += [new_results[i][0], int(100 * len(new_results[i]) / len(results) + 0.5)]
    return return_results

def main():
    logger = setup_logging({'log_to_console': True, 'log_to_file': False})
    f_path = r"C:\Users\SFZ Rechner\Downloads\output_2025-04-19-19-23-31_0_22.wav"
    m_path = r'C:\Users\SFZ Rechner\PycharmProjects\Semi10\Sound_processing\modelle\model_torch_1.pth'
    bees = ['no_event', 'swarm_event']
    model_text = 'l; conv2d; (1, 16); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (16, 48); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (48, 48); (3, 3); 1; (1, 1);; p; maxpool; (3, 3); 1; (1, 1);; v: view;; l; linear; (307200, 10);; l; linear; (10, 10);; l; linear; (10, 2);;'
    results = []
    path = r'F:\Aufnahmen\Fuchsfarm\6-10_Mai'
    paths = os.path.join(r'F:\Aufnahmen\Fuchsfarm\6-10_Mai', '*.flac')
    for i in range(len(os.listdir(path))):
        if os.path.isfile(p := os.path.join(path, os.listdir(path)[i])):
            if p.endswith('1.flac'):
                results.append(x:=[os.path.basename(p)[7:-5],predict(p, m_path, bees, logger, model_text, channel = 1)])
    os.chdir('..')
    with open(r'results/results_ff1_1.txt', 'a') as f:
        for i in results:
            f.write(str(i)+'\n')
        f.write('-------------------------\n')


    print(predict(f_path, m_path, bees, logger, model_text))

if __name__ == '__main__':
    main()

