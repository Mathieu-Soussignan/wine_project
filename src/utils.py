import os

def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)