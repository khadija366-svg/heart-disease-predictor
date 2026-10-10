"""
Heart Disease Prediction System - Flask Web App (PythonAnywhere version)
-------------------------------------------------------------------------
Same prediction logic as the Streamlit app, written as a plain Flask app so
it can run on PythonAnywhere (which hosts Flask / WSGI apps).

Run locally:   python flask_app.py    then open http://127.0.0.1:5000
PythonAnywhere: point the WSGI file to:  from flask_app import app as application
"""

from flask import Flask, render_template_string, request

import heart_model as hm

app = Flask(__name__)

# Load everything once when the app starts
try:
    MODEL, STATUS = hm.load_model()
    ENCODERS = hm.build_encoders()
    ACCURACY = hm.test_accuracy(MODEL)
    LOAD_ERROR = None
except Exception as load_problem:
    MODEL, STATUS, ENCODERS, ACCURACY = None, None, None, None
    LOAD_ERROR = (
        "The model could not be loaded or re-trained: " + str(load_problem)
        + ". Make sure svc_trained_model.pkl and training-data-encoded.csv "
        "are in the same folder as flask_app.py."
    )

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart Disease Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 20px; }
    .card { max-width: 560px; margin: 0 auto; background: #fff; padding: 24px;
            border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
    h1 { margin-top: 0; font-size: 1.6em; }
    label { display: block; margin-top: 14px; font-weight: bold; }
    select, button { width: 100%; padding: 10px; margin-top: 6px; font-size: 1em; }
    button { background: #c0392b; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
    .result { margin-top: 20px; padding: 14px; border-radius: 6px; font-weight: bold; text-align: center; }
    .bad { background: #fdecea; color: #b71c1c; }
    .good { background: #e8f5e9; color: #1b5e20; }
    .warn { background: #fff8e1; color: #6d4c00; padding: 10px; border-radius: 6px; margin: 14px 0; font-size: .9em; }
    .small { color: #666; font-size: .85em; margin-top: 18px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>&#10084;&#65039; Heart Disease Prediction System</h1>
    <p>Choose the four patient details and the trained Support Vector Classifier
       (SVC) will predict whether heart disease is likely.</p>
    {% if accuracy is not none %}
      <p><em>Model accuracy on held-out test data: <b>{{ "%.0f"|format(accuracy * 100) }}%</b></em></p>
    {% endif %}
    <div class="warn">Educational demo only. This is NOT medical advice.</div>

    <form method="post">
      {% for name, label in fields %}
        <label for="{{ name }}">{{ label }}</label>
        <select name="{{ name }}" id="{{ name }}">
          {% for option in categories[name] %}
            <option value="{{ option }}" {% if chosen[name] == option %}selected{% endif %}>{{ option }}</option>
          {% endfor %}
        </select>
      {% endfor %}
      <button type="submit">Predict Heart Disease</button>
    </form>

    {% if error %}
      <div class="result bad">{{ error }}</div>
    {% elif result is not none %}
      <div class="result {{ 'bad' if prediction == 1 else 'good' }}">{{ result }}</div>
    {% endif %}

    <p class="small">Built for teaching purposes from the UCI Heart Disease dataset
       (Cleveland) using only 4 input attributes.</p>
  </div>
</body>
</html>
"""

FIELDS = [
    ("ChestPain", "Chest Pain Type"),
    ("ExerciseAngina", "Exercise-Induced Angina"),
    ("MajorVessels", "Major Vessels Coloured by Fluoroscopy (0-3)"),
    ("Thalassemia", "Thalassemia (Thallium Stress Test)"),
]


@app.route("/", methods=["GET", "POST"])
def home():
    chosen = {name: hm.CATEGORIES[name][0] for name, _ in FIELDS}
    result, prediction, error = None, None, None

    if LOAD_ERROR:
        error = LOAD_ERROR
    elif request.method == "POST":
        for name, _ in FIELDS:
            chosen[name] = request.form.get(name, chosen[name])
        try:
            prediction, _ = hm.predict(MODEL, chosen, ENCODERS)
            result = hm.RESULT_TEXT[prediction]
        except Exception as problem:
            error = "Sorry, the prediction could not be made: " + str(problem)

    return render_template_string(
        PAGE,
        fields=FIELDS,
        categories=hm.CATEGORIES,
        chosen=chosen,
        result=result,
        prediction=prediction,
        error=error,
        accuracy=ACCURACY,
    )


if __name__ == "__main__":
    app.run(debug=True)
