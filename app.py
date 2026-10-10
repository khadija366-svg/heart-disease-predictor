"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
A small web page where anyone can pick 4 patient details and get a
prediction from the trained Support Vector Classifier (SVC).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (main file = app.py)

The model, encoders and fallback logic live in heart_model.py.
"""

import streamlit as st

import heart_model as hm

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_everything():
    """Load the model once and keep it in memory (fast for every visitor)."""
    model, status = hm.load_model()
    return {
        "model": model,
        "status": status,
        "encoders": hm.build_encoders(),
        "accuracy": hm.test_accuracy(model),
    }


try:
    state = load_everything()
except Exception as load_error:
    st.title("❤️ Heart Disease Prediction System")
    st.error(
        "The model could not be loaded or re-trained: " + str(load_error)
        + ". Make sure svc_trained_model.pkl and training-data-encoded.csv "
        "are uploaded in the same folder as app.py."
    )
    st.stop()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction System**
    notebook (UCI Heart Disease dataset, Cleveland data). Choose the four
    patient details below and the trained Support Vector Classifier (SVC)
    will predict whether heart disease is likely.
    """
)
if state["accuracy"] is not None:
    st.markdown(f"*Model accuracy on held-out test data: **{state['accuracy'] * 100:.0f}%***")

st.warning(
    "Educational demo only. This is NOT medical advice and must never be used "
    "to diagnose or treat a real patient."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type", hm.CATEGORIES["ChestPain"], index=3,
        help="Type of chest pain the patient reports.",
    )
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", hm.CATEGORIES["ExerciseAngina"],
        help="Does exercise cause chest pain (angina)?",
    )
with col2:
    major_vessels = st.selectbox(
        "Major Vessels Coloured by Fluoroscopy", hm.CATEGORIES["MajorVessels"],
        help="Number of major blood vessels (0 to 3) coloured by fluoroscopy.",
    )
    thalassemia = st.selectbox(
        "Thalassemia (Thallium Stress Test)", hm.CATEGORIES["Thalassemia"],
        help="Result of the thallium stress test.",
    )

if st.button("Predict Heart Disease", type="primary"):
    patient = {
        "ChestPain": chest_pain,
        "ExerciseAngina": exercise_angina,
        "MajorVessels": major_vessels,
        "Thalassemia": thalassemia,
    }
    try:
        prediction, encoded = hm.predict(state["model"], patient, state["encoders"])
        st.divider()
        if prediction == 1:
            st.error("### ⚠️ " + hm.RESULT_TEXT[1])
        else:
            st.success("### ✅ " + hm.RESULT_TEXT[0])
        with st.expander("See the encoded feature vector sent to the model"):
            st.dataframe(encoded, hide_index=True)
    except Exception as error:
        st.error("Sorry, the prediction could not be made: " + str(error))

if state["status"].startswith("Saved model could not be used"):
    st.info(state["status"])

st.divider()
st.caption(
    "Built for teaching purposes from the UCI Heart Disease dataset (Cleveland, "
    "297 complete patient records) using only 4 input attributes. Predictions "
    "reflect patterns in this small dataset, not a medical diagnosis."
)
