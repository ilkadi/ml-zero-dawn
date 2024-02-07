import yaml
import pprint
import os

class ConfigHelper:    
    def update_with_config(self, config_path, base_config={}):
        print(f"Loading config {config_path} ..")
        if config_path is not None:
            config_ext_path = os.path.join(config_path)
            with open(config_ext_path, 'r') as file:
                base_config.update(yaml.safe_load(file))    
        return base_config
    
    def print_config(self, config):
        print("Config:")
        pprint.pprint(config) 