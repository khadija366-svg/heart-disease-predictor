"""
heart_model.py - shared helper for the Heart Disease Prediction web apps
-------------------------------------------------------------------------
Both web apps (app.py for Streamlit and flask_app.py for PythonAnywhere)
use this one file, so the prediction logic is written only once.

What it does:
1. Keeps the list of the 4 input attributes and their allowed values.
2. Re-creates the same LabelEncoders that were trained in the notebook
   (LabelEncoder gives codes in alphabetical order, so fitting on the same
   fixed lists always gives the same numbers).
3. Loads the trained model from svc_trained_model.pkl.
4. FALLBACK: if the .pkl file is missing, damaged, or was saved with a
   different scikit-learn version, the model is re-trained automatically
   from training-data-encoded.csv (the same training data, same settings),
   so the app never crashes because of a model-file problem.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# All files live in the same folder as this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "svc_trained_model.pkl"
TRAIN_FILE = BASE_DIR / "training-data-encoded.csv"
TEST_FILE = BASE_DIR / "testing-data-encoded.csv"

# The 4 input attributes (exactly the same names and order as the notebook)
FEATURES = ["ChestPain", "ExerciseAngina", "MajorVessels", "Thalassemia"]
TARGET = "HeartDisease"

# Allowed values of every input attribute (same lists as notebook Step 5.1)
CATEGORIES = {
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    "ExerciseAngina": ["No", "Yes"],
    "MajorVessels": ["Zero", "One", "Two", "Three"],
    "Thalassemia": ["Normal", "Fixed Defect", "Reversible Defect"],
}
TARGET_CATEGORIES = ["Yes", "No"]

# What the numeric prediction means (the LabelEncoder gives No = 0, Yes = 1)
RESULT_TEXT = {1: "HEART DISEASE LIKELY", 0: "NO HEART DISEASE PREDICTED"}


def build_encoders():
    """Return one trained LabelEncoder per attribute (inputs and output)."""
    encoders = {name: LabelEncoder().fit(values) for name, values in CATEGORIES.items()}
    encoders[TARGET] = LabelEncoder().fit(TARGET_CATEGORIES)
    return encoders


def retrain_model():
    """Train the SVC again from the saved training data (fallback)."""
    training_data = pd.read_csv(TRAIN_FILE)
    input_vectors = training_data[FEATURES]
    output_labels = training_data[TARGET]
    model = svm.SVC(gamma="auto", random_state=0)
    model.fit(input_vectors, np.ravel(output_labels))
    return model


class ModelVersionMismatch(Exception):
    """The saved model was created with a different scikit-learn version."""


def load_model():
    """
    Load the saved model. Returns (model, status_text).
    If loading fails for ANY reason - or the file was saved with a different
    scikit-learn version, which can silently give wrong answers - fall back to
    re-training the same model from the saved training data.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(MODEL_FILE, "rb") as file:
                model = pickle.load(file)
        if any(type(item.message).__name__ == "InconsistentVersionWarning" for item in caught):
            raise ModelVersionMismatch("model saved with another scikit-learn version")
        # Quick self-test: make sure the loaded model really works
        model.predict(pd.DataFrame([[0, 0, 0, 0]], columns=FEATURES))
        return model, "Model loaded from svc_trained_model.pkl"
    except Exception as error:
        model = retrain_model()
        return model, (
            "Saved model could not be used (" + type(error).__name__ + "), "
            "so it was re-trained from training-data-encoded.csv"
        )


def encode_inputs(values, encoders=None):
    """Turn {'ChestPain': 'Asymptomatic', ...} into a 1-row numeric DataFrame."""
    encoders = encoders or build_encoders()
    missing = [name for name in FEATURES if name not in values]
    if missing:
        raise ValueError("Missing input(s): " + ", ".join(missing))
    row = {}
    for name in FEATURES:
        value = values[name]
        if value not in CATEGORIES[name]:
            raise ValueError(
                "Invalid value " + repr(value) + " for " + name + ". Allowed values: " + ", ".join(CATEGORIES[name])
            )
        row[name] = [int(encoders[name].transform([value])[0])]
    return pd.DataFrame(row, columns=FEATURES)


def predict(model, values, encoders=None):
    """Return (prediction_code, encoded_feature_vector). 1 = heart disease, 0 = no."""
    encoded = encode_inputs(values, encoders)
    prediction = int(model.predict(encoded)[0])
    return prediction, encoded


def test_accuracy(model):
    """Accuracy of the model on testing-data-encoded.csv (None if unavailable)."""
    try:
        testing_data = pd.read_csv(TEST_FILE)
        predictions = model.predict(testing_data[FEATURES])
        return float(accuracy_score(testing_data[TARGET], predictions))
    except Exception:
        return None
