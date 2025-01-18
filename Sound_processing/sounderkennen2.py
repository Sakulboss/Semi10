import mel_spec_calculator as msc
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import D3Modularised as D3M

#Datenformat: (1, 64, 100, 1) - (Dateien, bins in mfcc, werte pro bin, werte als liste)

filepath = 'viele_sounds_geordnet/chainsaw/1-19898-A-41.wav'
model_path = 'full_model_3_new.keras'
class_labels = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw', 'chirping_birds', 'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring', 'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind']  # Passen Sie dies an Ihre Datensätze an


# 1. Load the trained model
def predict(filepath: str, model_path: str, class_labels: list, printing=True):
    model = load_model(model_path)
    if printing: print("Model loaded")

    mel_spec = msc.compute_mel_spec_for_audio_file(filepath)
    mel_spec = np.array([i[:100] for i in mel_spec]) #Da CNN nur 100 Werte nehmen kann

    if printing: print("Mel spectrogram: ", mel_spec.shape)

    mel_spec = np.expand_dims(mel_spec, axis=-1)  # werte als liste
    mel_spec = np.expand_dims(mel_spec, axis=0)  # dateien

    scaler = StandardScaler()

    original_shape = mel_spec[0].shape
    mel_spec_2d = mel_spec[0].reshape(-1, original_shape[-1])
    mel_spec_scaled = scaler.fit_transform(mel_spec_2d)
    mel_spec[0] = mel_spec_scaled.reshape(original_shape)


    # 4. Make a prediction
    if printing: print("Performing classification ...")
    predictions = model.predict(mel_spec)  # Predict probabilities for each class
    predicted_class = np.argmax(predictions)
    if printing: print("Predicted class: ", class_labels[predicted_class], predicted_class)
    return predicted_class

