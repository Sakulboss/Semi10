import sounddevice as sd
from scipy.io.wavfile import write
from time import *
import os

def get_new_filename(file_extension: str) -> str:
	count = len([i for i in os.listdir() if i.endswith(file_extension)])
	return f'output_{count}.{file_extension}'

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
print('Aufnahme gestartet.')
sleep(4)	
#sd.wait()  # Wait until recording is finished
print('Aufnahme fertig')
write(get_new_filename("wav"), fs, myrecording)  # Save as WAV file
#write(get_new_filename("npy"), fs, myrecording)  # Save as npy??? file
print('Programmende')

	
print('Jetzt reichst mir jetzt rede ich! ES IST OBST IM HAUS ')	
print("^ Solche Sätze denkt sich auch nur Lukas aus")