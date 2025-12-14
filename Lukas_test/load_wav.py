import librosa

file_content, _ = librosa.load(r"C:\Users\sdose\Documents\GitHub\Semi10\Malte\konrad_22_2025-05-10-20-54-43_14.wav", sr=None, mono=False)

with open('c_output.txt', 'r') as f:
    c_content = f.readlines()
c_content = [ float((i.replace('\n', '').split(':'))[1]) for i in c_content]

file_content = file_content[0]



for i in range(200):
    if file_content[i] != c_content[i]:
        print(i, f":  {(file_content[i] - c_content[i])}  :  {file_content[i]:2.6f} | {c_content[i]:2.6f}")
    pass