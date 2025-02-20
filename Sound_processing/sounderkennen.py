from Sound_processing.Mülleimer import mel_spec_calculator as msc
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler


#Datenformat: (1, 64, 100, 1) - (Dateien, bins in mfcc, werte pro bin, werte als liste)



# 1. Load the trained model
def predict(filepath: str, model_path: str, class_labels: list, printing=True):
    verb = 0
    if printing: print('Model will load.')
    model = load_model(model_path)
    if printing:
        print("Model loaded")
        verb = 2
    mel_spec = msc.compute_mel_spec_for_audio_file(filepath)
    if printing: print('Mel specs erstellt')
    results = []

    for mel in chunks(mel_spec, 100):

        if printing: print("Mel spectrogram: ", mel.shape)

        mel = np.expand_dims(mel, axis=-1)  # werte als liste
        mel = np.expand_dims(mel, axis=0)  # datei dimension

        scaler = StandardScaler()
        original_shape = mel[0].shape
        mel_2d = mel[0].reshape(-1, original_shape[-1])
        mel_scaled = scaler.fit_transform(mel_2d)
        mel[0] = mel_scaled.reshape(original_shape)
        results.append(predict_result(mel, model, verb=verb))

    index = calculate_highest(results)
    return class_labels[index], index

def predict_result(mel_spec, model, verb=0):
    predictions = model.predict(mel_spec, verbose=verb)
    return np.argmax(predictions)

def chunks(l, n):
    """Yield successive n-sized chunks from sublists of l."""
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
    results = [int(i) for i in results]
    results.sort()
    sublists = {}

    if len(results) == 2: return results[0]

    for item in results:
        if item not in sublists:
            sublists[item] = []
        sublists[item].append(item)

    new_results = list(sublists.values())
    new_results.sort(key=len, reverse=True)

    return new_results[0][0]

if __name__ == '__main__':
    #filepath = 'viele_sounds_geordnet/chainsaw/1-19898-A-41.wav'
    filepath = 'viele_sounds_geordnet/cat/1-47819-A-5.wav'
    #filepath = 'viele_sounds_geordnet/airplane/1-36929-A-47.wav'
    model_path = 'full_model_3_new.keras'
    class_labels_big = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw',
                    'chirping_birds', 'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow',
                    'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',
                    'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw',
                    'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig',
                    'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring',
                    'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops',
                    'wind']

    print(predict(filepath, model_path, class_labels_big, True))