# Weakly Supervised Detection
Weakly Supervised Localization

Login to the machine  
make a folder A where your repository will reside  
$ cd folder-A  
  
In folder-A  
$ git clone https://github.com/mehak-agr/wsl.git  

Make a docker  
$ sudo docker run --gpus all --ipc=host -it -v /home/ben.bearce/mnt/:/data --name b_wsl projectmonai/monai:latest  

Following steps happen inside the docker  

cd to your folder-A   
$ cd /data/2015P002510/...folder-A  

Install wsl as a package  
$ pip install -e wsl  

To train a model just use the following command  
$ wsl train --debug  

Your trained model will be here - /data/2015P002510/Mehak/git_wsl/models/  
RSNA Data is here - /data/2015P002510/Mehak/git_wsl/data/rsna/  
Both these locations can be edited in wsl/locations.py file  

To download RSNA data visit - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data  

Upcoming updates
- Support for other datasets
- Claassification models adapted for regression tasks
- Out of order detection support
