# Setup

---
### 1. Install dependencies:
The following packages are required to run the code. You can install them using pip. Make sure you have Python and pip installed on your system.
0. Install and update pip:
```bash
  python -m pip install --upgrade pip
```

1. Install torch
```bash
  pip install torch
```
2. Install numpy
```bash
  pip install numpy
```
3. Install tqdm
```bash
  pip install tqdm
```
4. Install librosa
```bash
  pip install librosa
```
### 1.1 Summed up:

```bash
  pip install torch
  pip install numpy
  pip install tqdm
  pip install librosa
```
---
# 2. Setting up config.json
## 2.1 Explanation of the config.json file:
### 1. *logger_settings*:
| Key            | Explanation                                                                                                                      |                       Standard value                       |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------:|
| log_level      | level of logger;<br/> 0  = not set,<br/> 10 = debug,<br/> 20 = info,<br/> 30 = warnings,<br/> 40 = errors                        |                             20                             |
| log_to_console | True: logs are displayed in console <br/> False logs aren't displayed:                                                           |                            true                            |
| log_to_file    | True: logs are saved in: *log_file* <br/> False: logs aren't saved                                                               |                           false                            |
| log_file       | path to the logging file                                                                                                         |                          not set                           |
| log_format     | format of *logger* <br/> asctime: time <br/> name: name of file <br/> levelname: *log_level* <br/> messages: message from logger | ```%(asctime)s - %(name)s - %(levelname)s - %(message)s``` |

### 2. *training_data*:
| Key                                      | Explanation                                                                                                                          | Standard value |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|:--------------:|
| training_files_storage_location          | Path to the unsorted training files                                                                                                  |   cwd/_bees    |
| sorted_files_storage_location            | Path to the sorted training files, needs to be parent folder of the folder containing the class folders, empty folder if not run yet |      cwd       |
| mel_ceps_storage_location                | where the mel cepstrograms should be stored                                                                                          |      cwd       |
| training_file_extensions                 | file extension of training files                                                                                                     |      flac      |
| size                                     | type of the dataset to be used, implemented are ESC50 (enviromental sound classification - 50 classes) and bees_1 (our files)        |     bees_1     |
| segment_length_frames                    | length of the mel cepstrograms                                                                                                       |      100       |
| segments_per_spectrogram                 | amount of cepstrograms extracted from each file                                                                                      |      10        |
| create_new                               | if true, create new mel specs <br/> (when running the script for the first time, this will be set true)                              |     false      |
| create_new_source                        | if true, sort training files again <br/> (when running the script for the first time, this will be set true)                         |     false      |
cwd = current working directory
### 3. *model_settings*
| Key                  | Explanation                                                                                                  |                                                                                                                                                            Standard value                                                                                                                                                             |
|----------------------|--------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| swarm_event_ratio    | ratio of the swarm event data in datasets<br/> best: 0.5                                                     |                                                                                                                                                                  0.5                                                                                                                                                                  |
| test_size            | size of the test data; train data size is: <br/> 1 - *test_size*                                             |                                                                                                                                                                  0.3                                                                                                                                                                  |
| batch_size           | number of samples to work through before updating model parameters -> set higher for better training results |                                                                                                                                                                  64                                                                                                                                                                   |
| learning_rate        | amount of change in response to error, closer to zero if it should be more precise, else bigger              |                                                                                                                                                                 0.01                                                                                                                                                                  |
| max_epochs           | maximum epochs to iterate through, set higher when using bigger batch and smaller learning_rate              |                                                                                                                                                                  10                                                                                                                                                                   |
| min_epoch            | minimum epochs to iterate through, if set lower than 5, training may be stopped to early -> bad accuracy     |                                                                                                                                                                   5                                                                                                                                                                   |
| train_once           | if true, model is trained once on the given structure                                                        |                                                                                                                                                                 True                                                                                                                                                                  |
| model_text           | model architecture as described in Netzstruktur.py, needed if tain_once ist set to *true*                    | ```l; conv2d; (1, 16); (3, 3); 1; (1, 1);; a; relu;; p; maxpool; (5, 5); 1; (2, 2);; l; conv2d; (16, 48); (3, 3); 1; (1, 1);; a; relu;; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (48, 48); (11, 21); 1; (5, 10);; p; avgpool; (5, 5); 1; (2, 2);; v: view;; l; linear; (307200, 10);; l; linear; (10, 10);; l; linear; (10, 2);;``` |
| server_url           | URL of the server to; the model is received and the result is sent for training                              |                                                                                                                                                                  ---                                                                                                                                                                  |
| model_structure_file | file location for file containing model-structure                      |                                                                                                                                                                  cwd                                                                                                                                                                  |         
| use_server           | if true, server will be used to get the model structure and the results will be sent back                    |                                                                                                                                                                 true                                                                                                                                                                  | 
cwd = current working directory
## 2.2 Summed up:
Here you can find an example of a config.json file. It is recommended to use this as a template for your own config.json file. The values can be changed to fit your needs, but the keys should not be changed. The config.json file is used to configure the logger, training data and model settings.
You still need to set the paths to the training files and the model structure file. The logger, training data and model settings can be changed to fit your needs, but the default values should work fine for most cases. 
```json
{
  "logger_settings": {
    "log_level": 20,
    "log_to_console": true,
    "log_to_file": false,
    "log_file": "",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "training_data": {
    "training_files_storage_location": "./_bees",
    "sorted_files_storage_location": "./",
    "mel_ceps_storage_location": "./",
    "training_file_extensions": "flac",
    "size": "bees_1",
    "segment_length_frames": 100,
    "segments_per_spectrogram": 10,
    "create_new": false,
    "create_new_source": false
  },
  "model_settings": {
    "swarm_event_ratio": 0.5,
    "test_size": 0.3,
    "batch_size": 64,
    "learning_rate": 0.01,
    "max_epochs": 10,
    "min_epoch": 5,
    "train_once": true,
    "model_text": "",
    "server_url": "---",
    "model_structure_file": "../files/_netstruct.txt",
    "use_server": false
  }
}
```

# Execution:
### 1. Run the code:
To run the code, you need to have Python installed on your system. Once you have Python installed, you can run the code by executing the following command in your terminal or command prompt:
```bash
  python _driver_cnn.py
```
Or just run the file in your IDE.
#### Note:
If the accuracy of 50% occurs, the training is stopped. This is because the model is not able to learn anything, possibly due to a too small dataset (increase "segments_per_spectrogram" in config.json) or a too high learning rate (decrease "learning_rate" in config.json).

### 2. Explanation of the code:
#### A short explanation for every function is given in the code files as docstrings.
_driver_cnn.py:
- main file to run the code
- contains the main function

_driver_mels.py:
- file to create mel cepstrograms

ccn_train_net.py:
- file to train the model

ccn_net_prep.py:
- file to prepare the model

ccn_data_prep.py:
- file to prepare the data

data_file_prep.py:
- file to prepare the data files

data_label.py: 
- file to label the data files
- class names and labels are defined here

data_mel_cepstrograms.py:
- file to create mel cepstrograms from all selected data files

data_refining.py:
- shapes the data to fit the model

# Used packages:
The following packages were not created by the authors of this code, but are used to run the code.
Not Python native packages are:
- numpy
- librosa
- torch 
- tqdm

Python native packages are:
- glob
- json
- logging
- os
- shutil

