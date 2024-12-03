import wave, struct
import matplotlib.pyplot as plt

wavefile = wave.open(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav", 'r')

length = wavefile.getnframes()

numbers = 1
liste123 = []
for i in range(0, length):
    #wavedata = wavefile.readframes(1)
    #data = struct.unpack("<h", wavedata)
    
    wavedata = wavefile.readframes(numbers)
    data = struct.unpack(f"<{numbers}h", wavedata)
    
    liste123.append(int(data[0]))

#This snippet reads 1 frame. To read more than one frame (e.g., 13), use
plt.title("Signal Wave...")
plt.plot(liste123)
plt.show()
