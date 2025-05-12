import soundfile as sf

# Lese die Stereo WAV-Datei
data, samplerate = sf.read("konrad_17_2_2025-05-10-20-40-39.wav")

# Überprüfe, ob es sich um Stereo handelt
if data.ndim == 2:
    # Teile die Stereo-Daten in zwei Mono-Daten auf
    left_channel = data[:, 0]
    right_channel = data[:, 1]

    # Speichere die Mono-Daten in neuen WAV-Dateien
    sf.write("mono_left1.wav", left_channel, samplerate)
    sf.write("mono_right1.wav", right_channel, samplerate)
