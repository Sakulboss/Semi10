import sounddevice as sd
from scipy.io.wavfile import write
from time import *
from datetime import datetime
import os

def get_new_filename(file_extension: str) -> str:
	count = len([i for i in os.listdir() if i.endswith(file_extension)])
	return f'output_{str(datetime.now()).replace(" ", "-").replace(".", "-")}_{count}.{file_extension}'

def pathway():
	pass

def aufnahmen(seconds: float, fs = 44100, printRecording = True) -> None:
	myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
	print('Aufnahme gestartet.')
	sleep(seconds + 1)
	#sd.wait()  # Wait until recording is finished
	print('Aufnahme fertig')
	if printRecording:
		print(myrecording)
	return myrecording

def speichern(myrecording, fs =  44100) -> None:
	write(get_new_filename("wav"), fs, myrecording)  # Save as WAV file
	print('Programmende')

def main():
	dauer = 3
	speichern(aufnahmen(dauer))

if __name__ == '__main__':
	#main()
	# fs: Sample rate
	# seconds: Duration of recording
