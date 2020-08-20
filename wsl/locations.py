'''This is a module for including project wide definitions of locations where files are stored.'''
import json
from pathlib import Path
import subprocess
import sys
from os import environ

from pkg_resources import resource_filename


# Standard location for share mounting
# This will vary based on the platform
if sys.platform == 'darwin':
    # 'darwin' just means MacOS
    project_share_mount = Path('/Volumes/2015P002510')
else:
    # Linux servers and containers
    project_share_mount = Path('/data/2015P002510')

# Issue a warning here if the project share doesn't exist
if not project_share_mount.exists() and not cluster_utils.is_slurm_job():
    raise FileNotFoundError(
        f'The project share is not mounted at the expected location {project_share_mount}'
    )

# Part of the share dedicated to the object detection subproject
project_root = project_share_mount / 'Mehak' / 'git_wsl'

# Location of available datasets
project_data_dir = subproject_root / 'data'

# Location of model dir where checkpoints are stored
project_model_dir = subproject_root / 'models'

# Location where train configs are kept
train_configs_dir = Path(resource_filename('wsl', 'configs'))