#!/usr/bin/python3
from pathlib import Path

user = 'mehak'
known_extensions = {'rsna': 'dcm', 'chexpert': 'jpg', 'chestxray8': 'png', 'siim': 'dcm'}
known_tasks = {'rsna': 'detect', 'siim': 'segment'}

# root = Part of the share dedicated to the object detection subproject
# wsl_data_dir = Location of available datasets
# wsl_csv_dir = Location of csvs
# wsl_model_dir = Location of model dir where checkpoints are stored
# wsl_summary_dir = Location of model dir where checkpoints are stored

if user == 'mehak':
    root = Path('/data/2015P002510/Mehak/git_wsl')
    wsl_data_dir = root / 'data'
    wsl_csv_dir = root / 'wsl' / 'wsl' / 'csvs'
    wsl_model_dir = root / 'models'
    wsl_summary_dir = root / 'summary'
    wsl_plot_dir = root / 'plots'
else:
    print(f'User {user} not recognized.')

'''

How to use?

Please use <user> variable to reflect your name
todo: make setting this a part of docker setup

Add to the if-else bock to add the location of storage - maintainer recommends using full paths to avoid confusion
Needless go say, do not edit somebody else's data directory paths

The paths added here can be called all over the repo, do not use actual paths anywhere else

Data directory structure:
Each image location is presumed to be as follows:
<wsl_data_dir> / <data> / <id>.<extension>
where,
    <data> is specified as argument during run time itself - currently supports rsna and chexpert
    <id> each unique study or image is accessed by this label - first column of each csv is Id - confirm in wsl_csv_dir
    <extension> is also specified as argument during runtime - it is decoupled from id since each extension requires slightly different loading function

CSV directory structure:
<wsl_csv_dir> / <data> / <file_name>.csv
Each csv directory 4 main files - info, train, valid, test
info.csv - first column is Id, the rest are ground truths
         - 0/1 for binary classification, simple integers for regression
         - for binary classification if the 0 and 1 are represent different characterstics, name the column <0>_<1>
            e.g. Male_Female where 0 stands for Male and 1 for Female in the column
         - for a > 1 classes, make a single column with a list of ground-truth labels for each Id
         - todo: include support for multi-class regression and multi-class + multi-label ground truths
train.csv, valid.csv test.csv - single column csv which contains ids to use for current data split
original.csv - (optional) csv from which info.csv was derived

Model directory structure:
Please refer to wsl / wsl / train.py for how the models are named

Summary directory structure:
Used for storing combined results upon testing of all models - ease of comparison for different architectures and variants

'''
