# Heart Disease Prediction System - Web App

A small web app that predicts whether heart disease is likely from **4 input attributes**
(Chest Pain Type, Exercise-Induced Angina, Major Vessels, Thalassemia), using a Support
Vector Classifier trained on the UCI Heart Disease dataset (Cleveland data).

> Educational demo only. Not medical advice.

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (main file for Streamlit Community Cloud) |
| `flask_app.py` | Same app written in Flask (for PythonAnywhere) |
| `heart_model.py` | Shared code: encoders, model loading, prediction, **automatic re-training fallback** |
| `svc_trained_model.pkl` | Trained model saved by the notebook |
| `training-data-encoded.csv` | Training data (used only if the model has to be re-trained) |
| `testing-data-encoded.csv` | Test data (used to show the accuracy on the page) |
| `requirements.txt` | Python packages for deployment |

## Run on your own computer first

```
pip install -r requirements.txt
streamlit run app.py
```
It opens at http://localhost:8501

## Option A (recommended): Streamlit Community Cloud (free)

1. Create a **public** GitHub repository and upload everything in this folder (files at the top level, not inside a sub-folder).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> **Deploy a public app from GitHub**.
4. Repository = your repo, Branch = `main`, Main file path = `app.py`.
5. Click **Deploy** and wait a few minutes. Copy your `https://<name>.streamlit.app` link.
6. If the build fails, open **Manage app -> Settings -> Python version** and pick **3.11** or **3.12**, then reboot.

## Option B (fallback): PythonAnywhere (Flask)

1. Sign up at https://www.pythonanywhere.com (free account) and open the **Files** tab.
2. Create a folder `heart-disease-app` and upload every file from this folder into it.
3. Open the **Web** tab -> **Add a new web app** -> **Flask** (pick a Python version).
4. Open the WSGI configuration file and replace its content with:
   ```python
   import sys
   path = "/home/YOUR_USERNAME/heart-disease-app"
   if path not in sys.path:
       sys.path.insert(0, path)
   from flask_app import app as application
   ```
5. In a **Bash console** check the packages: `python -c "import sklearn, pandas"` (if missing: `pip install --user scikit-learn pandas`).
6. Press **Reload** on the Web tab. Your link is `https://YOUR_USERNAME.pythonanywhere.com`.

(Follow your instructor's PythonAnywhere tutorial for the exact clicks if the screens look different.)

## Option C (fallback): Hugging Face Spaces

Create a new Space, choose the **Streamlit** SDK, upload the same files, and keep `app.py` as the app file.

## If the model file ever fails to load

`heart_model.py` catches the error and re-trains the same SVC from `training-data-encoded.csv`
automatically, so the app keeps working (for example after a scikit-learn version change).
