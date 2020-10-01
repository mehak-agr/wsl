# Weakly Supervised Localization

## Directory Structure

### Data
Each image location is presumed to be as follows: <br />
wsl_data_dir / data / id.extension <br />
where, <br />
* data is specified as argument during run time itself - currently supports rsna and chexpert <br />
* id each unique study or image is accessed by this label - first column of each csv is Id - confirm in wsl_csv_dir <br />
* extension is also specified as argument during runtime - it is decoupled from id since each extension requires slightly different loading function <br />
        
### CSVs
wsl_csv_dir / data / file_name.csv <br />
Each csv directory 4 main files - info, train, valid, test <br />
info.csv - first column is Id, the rest are ground truths <br />
* 0/1 for binary classification, simple integers for regression <br />
* for binary classification if the 0 and 1 are represent different characterstics, name the column 0_1 <br />
  e.g. Male_Female where 0 stands for Male and 1 for Female in the column <br />
* for a > 1 classes, make a single column with a list of ground-truth labels for each Id <br />
* todo: include support for multi-class regression and multi-class + multi-label ground truths <br />
train.csv, valid.csv test.csv - single column csv which contains ids to use for current data split <br />
original.csv - (optional) csv from which info.csv was derived <br />
    
### Model
Please refer to wsl / wsl / train.py for how the models are named <br />

### Summary
Used for storing combined results upon testing of all models - ease of comparison for different architectures and variants <br />

## What to change?

Go to wsl / locations.py
Use user variable to reflect your name <br />
Add to the if-else bock to add the location of storage - maintainer recommends using full paths to avoid confusion <br />
The paths added here can be called all over the repo, do not use actual paths anywhere else <br />
TODO: make setting this a part of docker setup <br />

## How to run?

Login to the machine  <br />

### Occurs on your remote / local machine
Make a folder A where your repository will reside  - $ mkdir folder-A  <br />
Go to foler-A - $ cd folder-A  <br />
Inside folder-A  - $ git clone https://github.com/mehak-agr/wsl.git  <br />
Make a docker - $ sudo docker run --gpus all --ipc=host -it -v /home/mehak.aggarwal/mnt/:/data --name m_wsl projectmonai/monai:latest  <br />

### Occurs inside docker m_wsl
Go to folder-A - $ cd /data/2015P002510/...folder-A  <br />
Install wsl as a package  - $ pip install -e wsl  <br />
To train a debug model to ensure things work fine  - $ wsl medinet --debug  <br />
<br />
You can find a comprehensive list of commands and arguments in wsl/main.py  <br />
Your trained model will be here - wsl_model_dir / models /  <br />

## Upcoming updates
- UNet for segmentation
