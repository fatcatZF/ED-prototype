import os 
from joblib import load 
import json 


def load_lof(env_name, load_norm_config=True):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "ed-models", env_name.lower(), "lof", "pipeline_lof_0.joblib")
    config_path = os.path.join(base_dir, "ed-models", env_name.lower(), "lof", "norm_config_lof_0.json")


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for environment: {env_name}")
    
    if load_norm_config and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return load(model_path), config
    

    return load(model_path)



