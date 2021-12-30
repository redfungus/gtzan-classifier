# GTZAN CNN Classifier

This repo contains a simple implementation of a CNN for classifying music genres. It is implemented in Pytorch and Pytorch Lightning. The repo also comes with a VSCode .devcontainer configuration to allow for easy development using VSCode. It was implemented as a homework for an interview.

### Installation
Either open Dev container using VSCode to have all requirements installed automatically or manually install packages using `pip install -r requirements.txt`.

### Download Dataset and pre-process
You can download the dataset and perform initial pre-processing using `make download-dataset`. After this step run `python scripts/prepare_data.py` to pre-process the data to be used by the model. You can use the `--augment-dataset` flag to enable data augmentation. You can control the number of augmentations per wav file using `--augments-per-sample`.

### Structure
All scripts are located in the `scripts` folder. The data will be downloaded to `.data`. 
Below, you can see a small description of each file:

- `create_dataset_csv.py`: This file contains the script used for generating a csv file of the paths of the wav files from the original dataset. It is ran automatically with `make download-dataset`.
- `data_exploration.ipynb`: This notebook contains code for doing data exploration on the data.
- `gtzan_dataset.py`: This file contains the custom Pytorch DataSet class which is used by the model.
- `models.py`: This file contains the implementation of the GTZAN classifier model.
- `prepare_data.py`: This file contains the methods for converting wav files to mel-spectograms alongside a simple pipeline for processing the data to be used by the model.
- `train.ipynb`: This notebook contains the main training loop which used the other classes to train a model and evaluate it.
- `vscode_audio.py`: This file contains a workaround to enable playing wav files in VSCode notebooks. (not important)

### Requirements
* VSCode
* Docker
* VSCode plugins needed for remote development. For more information visit [here](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl).

