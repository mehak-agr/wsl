'''This is a module for including project wide definitions of locations where files are stored.'''
from pathlib import Path


# Standard location for share mounting
mount = Path('/data/2015P002510')

# Issue a warning here if the project share doesn't exist
if not mount.exists():
    raise FileNotFoundError(
        f'The project share is not mounted at the expected location {mount}'
    )

# Part of the share dedicated to the object detection subproject
root = mount / 'Mehak' / 'git_wsl'

# Location of available datasets
wsl_data_dir = root / 'data'

# Location of model dir where checkpoints are stored
wsl_model_dir = root / 'models'
