import librosa

file = librosa.load('Bienen_2_channel.wav', sr=None, mono=False)
print(file[0][0])
print(file[0][1])