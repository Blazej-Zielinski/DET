import os
from src.config import Config


def create_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_file_name(config: Config, folder_path):
    return os.listdir(folder_path)
