import sounddevice as sd
from scipy.io.wavfile import write
from time import *

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
print('Programmene')
sleep(4)	
#sd.wait()  # Wait until recording is finished
print('Programmene')
write('output.wav', fs, myrecording)  # Save as WAV file
write('output1.npy', fs, myrecording)  # Save as npy??? file
print('Programmene')	
with open("output2.npy", "w+") as file:
	file.write(str(myrecording))
with open("output3.wav", "w+") as file:
	file.write(str(myrecording))
	
print('Jetzt reichst mir jetzt rede ich! ES IST OBST IM HAUS ')	
