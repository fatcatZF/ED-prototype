import os 
from joblib import load 


def load_lof(env_name):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "ed-models", env_name.lower(), "pipeline_lof_0.joblib")


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for environment: {env_name}")
    

    return load(model_path)



