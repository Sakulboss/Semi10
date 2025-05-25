# Setup:
### 1. Install dependencies:

---
1. Install torch
```
pip install torch
```
2. Install numpy
```
pip install numpy
```
3. Install tqdm
```
pip install tqdm
```
4. Install librosa
```
pip install librosa
```
### 1.1 Summed up:

```
pip install torch
pip install numpy
pip install tqdm
pip install librosa
```
# 2. config.json explained:

---

### 1. *logger_settings*:
| Key            | Explanation                                                                                               | Standard value |
|----------------|-----------------------------------------------------------------------------------------------------------|----------------|
| log_level      | level of logger;<br/> 0  = not set,<br/> 10 = debug,<br/> 20 = info,<br/> 30 = warnings,<br/> 40 = errors |                |
| log_to_console | True: <br/> False:                                                                                        |                |
| log_to_file    | True: logs are saved in: *log_file* <br/> False: logs aren't saved                                        |                |
| log_file       | path to the logging file                                                                                  |                |
| log_format     | format of *logger* <br/> use: ```%(asctime)s - %(name)s - %(levelname)s - %(message)s```                  |                |

### 2. *training_data*:
| Key                             | Explanation | Standard value |
|---------------------------------|-------------|----------------|
| create_new_source               |             |                |
| training_files_storage_location |             |                |
| sorted_files_storage_location   |             |                |
| mel_specs_storage_location      |             |                |
| training_file_extensions        |             |                |
| size                            |             |                |
| segment_length_frames           |             |                |
| create_new                      |             |                |
| printing                        |             |                |

### 3. *model_settings*
| Key               | Explanation                                              | Standard value |
|-------------------|----------------------------------------------------------|----------------|
| swarm_event_ratio | ratio of the swarm event data in datasets<br/> best: 0.5 |                |
| test_size         |                                                          |                |
| batch_size        |                                                          |                |
| learning_rate     |                                                          |                |
| max_epochs        |                                                          |                |
| min_epoch         |                                                          |                |
| train_once        |                                                          |                |
| model_text        |                                                          |                |
| dropbox           |                                                          |                |
| printing          |                                                          |                |


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
