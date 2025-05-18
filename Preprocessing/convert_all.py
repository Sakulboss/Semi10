import os
import time
from pydub import AudioSegment

def convert_wav_to_flac_and_delete_old(folder_path):
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found in the folder.")
        return

    start_time = time.time()
    for index, filename in enumerate(wav_files, start=1):
        wav_file_path = os.path.join(folder_path, filename)
        flac_file_path = os.path.join(folder_path, filename[:-4] + '.flac')

        # Load the .wav file
        audio = AudioSegment.from_wav(wav_file_path)

        # Export as .flac
        audio.export(flac_file_path, format='flac')
        print(f'Converted: {filename} to {filename[:-4]}.flac')

        # Delete the original .wav file
        os.remove(wav_file_path)
        print(f'Deleted original file: {filename}')

        # Calculate and print progress and estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / index
        files_left = total_files - index
        est_time_left = avg_time_per_file * files_left/60
        print(f'Files left: {files_left}')
        print(f'Estimated time remaining: {est_time_left:.2f}min\n')

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .wav files: ")
    convert_wav_to_flac_and_delete_old(folder_path)