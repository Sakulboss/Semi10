# Setup:
### 1. Install dependencies:
The following packages are required to run the code. You can install them using pip. Make sure you have Python and pip installed on your system.


---
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
# 2. config.json explained:

---

### 1. *logger_settings*:
| Key            | Explanation                                                                                                                      |                       Standard value                       |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------:|
| log_level      | level of logger;<br/> 0  = not set,<br/> 10 = debug,<br/> 20 = info,<br/> 30 = warnings,<br/> 40 = errors                        |                             20                             |
| log_to_console | True: logs are displayed in console <br/> False logs aren't displayed:                                                           |                            true                            |
| log_to_file    | True: logs are saved in: *log_file* <br/> False: logs aren't saved                                                               |                           false                            |
| log_file       | path to the logging file                                                                                                         |                          not set                           |
| log_format     | format of *logger* <br/> asctime: time <br/> name: name of file <br/> levelname: *log_level* <br/> messages: message from logger | ```%(asctime)s - %(name)s - %(levelname)s - %(message)s``` |

### 2. *training_data*:
| Key                             | Explanation                                                                                                                          | Standard value |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|:--------------:|
| training_files_storage_location | Path to the unsorted training files                                                                                                  |   cwd/_bees    |
| sorted_files_storage_location   | Path to the sorted training files, needs to be parent folder of the folder containing the class folders, empty folder if not run yet |      cwd       |
| mel_ceps_storage_location       | where the mel cepstrograms should be stored                                                                                          |      cwd       |
| training_file_extensions        | file extension of training files                                                                                                     |      flac      |
| size                            | type of the dataset to be used, implemented are ESC50 (enviromental sound classification - 50 classes) and bees_1 (our files)        |     bees_1     |
| segment_length_frames           | length of the mel cepstrograms                                                                                                       |      100       |
| create_new                      | if true, create new mel specs <br/> (when running the script for the first time, this will be set true)                              |     false      |
| create_new_source               | if true, sort training files again <br/> (when running the script for the first time, this will be set true)                         |     false      |
cwd = current working directory
### 3. *model_settings*
| Key               | Explanation                                                          | Standard value |
|-------------------|----------------------------------------------------------------------|:--------------:|
| swarm_event_ratio | ratio of the swarm event data in datasets<br/> best: 0.5             |      0.5       |
| test_size         | size of the test_data; train data results from:<br/> 1 - *test_size* |      0.3       |
| batch_size        |                                                                      |                |
| learning_rate     |                                                                      |                |
| max_epochs        |                                                                      |                |
| min_epoch         |                                                                      |                |
| train_once        |                                                                      |                |
| model_text        |                                                                      |                |
| dropbox           |                                                                      |                |
| printing          |                                                                      |                |


  - which file to run
  - config: what is needed to set correctly and what does it do
    - logger
    - data
    - model
    
- how program works:
  - data loading
  - model training
  - model evaluation
  - saving results



### Used packages:
- numpy
- torch 
- logging
- os
- json
- shutil
- glob
