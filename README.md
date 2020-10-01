# Weakly Supervised Detection
Weakly Supervised Localization

What to change?

Please use <user> variable to reflect your name <br />
todo: make setting this a part of docker setup <br />
Add to the if-else bock to add the location of storage - maintainer recommends using full paths to avoid confusion <br />
The paths added here can be called all over the repo, do not use actual paths anywhere else <br />
Data directory structure: <br />
Each image location is presumed to be as follows: <br />
<wsl_data_dir> / <data> / <id>.<extension> <br />
where, <br />
    <data> is specified as argument during run time itself - currently supports rsna and chexpert <br />
    <id> each unique study or image is accessed by this label - first column of each csv is Id - confirm in wsl_csv_dir <br />
    <extension> is also specified as argument during runtime - it is decoupled from id since each extension requires slightly different loading function <br />
CSV directory structure: <br />
<wsl_csv_dir> / <data> / <file_name>.csv <br />
Each csv directory 4 main files - info, train, valid, test <br />
info.csv - first column is Id, the rest are ground truths <br />
         - 0/1 for binary classification, simple integers for regression <br />
         - for binary classification if the 0 and 1 are represent different characterstics, name the column <0>_<1> <br />
            e.g. Male_Female where 0 stands for Male and 1 for Female in the column <br />
         - for a > 1 classes, make a single column with a list of ground-truth labels for each Id <br />
         - todo: include support for multi-class regression and multi-class + multi-label ground truths <br />
train.csv, valid.csv test.csv - single column csv which contains ids to use for current data split <br />
original.csv - (optional) csv from which info.csv was derived <br />
Model directory structure: <br />
Please refer to wsl / wsl / train.py for how the models are named <br />
Summary directory structure: <br />
Used for storing combined results upon testing of all models - ease of comparison for different architectures and variants <br />

How to run?

Login to the machine  
make a folder A where your repository will reside  
$ cd folder-A  
In folder-A  
$ git clone https://github.com/mehak-agr/wsl.git  
Make a docker  
$ sudo docker run --gpus all --ipc=host -it -v /home/mehak.aggarwal/mnt/:/data --name m_wsl projectmonai/monai:latest  

Following steps happen inside the docker  
cd to your folder-A   
$ cd /data/2015P002510/...folder-A  
Install wsl as a package  
$ pip install -e wsl  
To train a model just use the following command  
$ wsl medinet --debug  

Your trained model will be here - /data/2015P002510/Mehak/git_wsl/models/  
RSNA Data is here - /data/2015P002510/Mehak/git_wsl/data/rsna/  
Both these locations can be edited in wsl/locations.py file  

To download RSNA data visit - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data  

Upcoming updates
- UNet for segmentation
