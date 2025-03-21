from flask import Flask, request, jsonify

from .load_model import * 


import numpy as np 


app = Flask(__name__)

model_cache = {}




@app.route("/predict_drift_score", methods=["POST"]) 
def predict_drift_score():
    data = request.json

    # Check required fields
    if not all(k in data for k in ("t", "st", "at", "stp1", "env_name")):
        return jsonify({"error": "Missing one or more required fields: 't', 'st', 'at', 'stp1', 'env_name'"}), 400
    
    try:
        timestamp = data["t"]
        st = np.array(data["st"])
        at = np.array(data["at"])
        stp1 = np.array(data["stp1"])
        env_name = data["env_name"]

        transition = np.concatenate([st, stp1-st]).reshape(1, -1)
        x = np.concatenate([transition, at.reshape(1,-1)], axis=1).astype(np.float32)

        # load or cache model
        if env_name not in model_cache:
            ed_model, norm_config = load_lof(env_name)
            model_cache[env_name] = ed_model
            model_cache[f"{env_name}_norm_config"] = norm_config

        ed_model = model_cache[env_name]
        norm_config = model_cache[f"{env_name}_norm_config"]

        drift_score = (-ed_model.decision_function(x)[0]-norm_config["mu"])/(norm_config["std"]+1e-6)

        return jsonify({
           't': timestamp,
           "drift_score": drift_score,
           "env_name": env_name
        }), 200
    
    except Exception as e: 
        return jsonify({"error": f"Internal Error: {str(e)}"}), 500 
    


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)





