import json
import yaml
import os

class ConfigManager:
    DEFAULT_CONFIG = {
        "log_level": "INFO",
        "log_file": "app.log",
        "utterance_patterns": []
    }

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()

    def load_config(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The configuration file {file_path} does not exist.")

        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                self.config.update(json.load(file))
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                self.config.update(yaml.safe_load(file))
            else:
                raise ValueError("Unsupported file format. Please use JSON or YAML.")

    def merge_config(self, new_config):
        self.config.update(new_config)

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def set_config(self, key, value):
        self.config[key] = value
