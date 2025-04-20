import os

def soundpath(name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path[:-len(os.path.basename(dir_path))-1]
    dir_path = os.path.join(dir_path, '_sounddateien')
    dir_path = os.path.join(dir_path, name)
    return dir_path

if __name__ == '__main__':
    print(soundpath('test.wav'))

"""
from pathfinder import soundpath

file_dir_path = soundpath('test.wav')
"""