"""
Usage:
    python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"
"""

import argparse
import json
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

def load_target_names():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}. Please train the model first.")
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
    
    return data.get("target_names", [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, 
                        help="Feature List as JSON. Example: \"[5.1, 3.5, 1.4, 0.2]'")
    args = parser.parse_args()

    # Parse input
    try:
        input_features = json.loads(args.input)
        if not isinstance(input_features, list) or len(input_features) != 4:
            raise ValueError("Input must be a list of 4 numerical features.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for input features.")
    
    X = np.array(input_features).reshape(1, -1)

    print(f"Input features: {X}")
    model = load_model()
    pred = model.predict(X)

    target_names = load_target_names()
    print(f"prediction: {pred}")
    print(f"prediction: {target_names[pred[0]]}")


if __name__ == "__main__":
    main()