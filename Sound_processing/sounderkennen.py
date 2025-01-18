from tensorflow.keras.models import Sequential, load_model, Model
from D3Modularised import *
import mel_spec_calculator as msc

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import D3Modularised as msc  # Assuming your modularized code is imported here


def classify_audio_file(filepath, model_path, class_labels):
    """
    Classifies an audio file using a pre-trained model.

    Args:
        filepath (str): Path to the audio file to be classified.
        model_path (str): Path to the saved model.
        class_labels (list): List of class names (e.g., ["cat", "dog"]).

    Returns:
        str: The predicted class for the input audio file.
    """

    # 1. Load the trained model
    print("Loading the model ...")
    model = load_model(model_path)
    print("Model loaded.")

    # 2. Prepare input data
    print("Calculating the Mel spectrogram for the audio file ...")
    mel_spec = msc.compute_mel_spec_for_audio_file(filepath)

    # Check and expand dimensions to match model requirements
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension

    # 3. Normalize the data (reshape to apply scaler)
    print("Normalizing input data ...")
    scaler = StandardScaler()

    # Reshape mel_spec[0] to be 2D for scaling
    original_shape = mel_spec[0].shape  # Save shape for restoring later
    mel_spec_2d = mel_spec[0].reshape(-1, original_shape[-1])  # Reshape to 2D
    mel_spec_scaled = scaler.fit_transform(mel_spec_2d)  # Apply scaler
    mel_spec[0] = mel_spec_scaled.reshape(original_shape)  # Reshape back to 3D

    # 4. Make a prediction
    print("Performing classification ...")
    predictions = model.predict(mel_spec)  # Predict probabilities for each class
    predicted_class = np.argmax(predictions)  # Index of the highest probability

    # 5. Return the predicted class
    return class_labels[predicted_class]

'''print('Model started Loading!')
model = load_model('full_model_3_new.keras')
print('Model Loaded!')
printing = True
classes = model_evaluation(big=True,
                 printing=printing,
                 trained_model=model)'''

audio_file_path = "viele_sounds_geordnet/clock_tick/1-21934-A-38.wav"  # Pfad zu deiner Audiodatei
model_path = "full_model_3_new.keras"  # Pfad zu deinem gespeicherten Modell
class_labels = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw', 'chirping_birds', 'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring', 'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind']  # Passen Sie dies an Ihre Datensätze an

predicted_label = classify_audio_file(audio_file_path, model_path, class_labels)
print(f"Die Datei '{audio_file_path}' wurde als '{predicted_label}' klassifiziert.")
print("Done :)")