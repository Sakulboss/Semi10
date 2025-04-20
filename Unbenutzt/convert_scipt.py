from pydub import AudioSegment
import os


def cflac(wav_file_path):
	# Load the WAV file
	audio = AudioSegment.from_wav(wav_file_path)
	# Create the output file path by changing the extension to .flac
	flac_file_path = os.path.splitext(wav_file_path)[0] + '.flac'
	# Export as FLAC
	audio.export(flac_file_path, format="flac")
	print(f"Converted {wav_file_path} to {flac_file_path}")

# Convert FLAC to WAV
def cwav(flac_file_path):
	# Load the FLAC file
	audio = AudioSegment.from_file(flac_file_path, format="flac")
	# Create the output file path by changing the extension to .wav
	wav_file_path = os.path.splitext(flac_file_path)[0] + '.wav'
	# Export as WAV
	audio.export(wav_file_path, format="wav")
	print(f"Converted {flac_file_path} to {wav_file_path}")

# Splitting

def splitter(wav_file_path):
	# Load the stereo WAV file
	audio = AudioSegment.from_wav(wav_file_path)
	# Check if the audio is stereo
	if audio.channels != 2:
		raise ValueError("The input file is not a stereo WAV file.")
	# Split into left and right channels
	left_channel = audio.split_to_mono()[0]
	right_channel = audio.split_to_mono()[1]
	# Create output file paths
	base_name = os.path.splitext(wav_file_path)[0]
	left_channel_path = f"{base_name}_left.wav"
	right_channel_path = f"{base_name}_right.wav"
	# Export the mono channels as separate WAV files
	left_channel.export(left_channel_path, format="wav")
	right_channel.export(right_channel_path, format="wav")
	print(f"Split {wav_file_path} into {left_channel_path} and {right_channel_path}")

if __name__ == '__main__':
	cwav("")
	cwav("")