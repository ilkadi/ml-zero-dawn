import yaml
import pprint
import os

class ConfigHelper:    
    def update_with_config(self, config_path, base_config={}):
        if config_path is None or not os.path.exists(config_path):
            raise ValueError("Please provide a valid config path.")
        
        print(f"Loading config {config_path} ..")
        config_ext_path = os.path.join(config_path)
        with open(config_ext_path, 'r') as file:
            base_config.update(yaml.safe_load(file))  
        return base_config
    
    def print_config(self, config):
        print("Config:")
        pprint.pprint(config) 