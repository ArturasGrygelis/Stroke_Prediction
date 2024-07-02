import io
import pickle
import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


# Load the LightGBM model and other necessary objects
with open('lgb1_model.pkl', 'rb') as f:
    lgb1 = pickle.load(f)

categorical_features = joblib.load("categorical_features.joblib")
encoder = joblib.load("encoder.joblib")

# Sidebar option to select the dashboard
option = st.sidebar.selectbox("Which dashboard?", ("Model information", "Stroke prediction"))
st.title(option)

def get_pred():
    """
    Function to display the stroke probability calculator and Shap force plot.
    """
    st.header("Stroke probability calculator ")

    # User input for prediction
    gender = st.selectbox("Select gender: ", ["Male", "Female", 'Other'])
    work_type = st.selectbox("Work type: ", ["Private", "Self_employed", 'children', 'Govt_job', 'Never_worked'])
    residence_status = st.selectbox("Residence status: ", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking status: ", ["Unknown", "formerly smoked", 'never smoked', 'smokes'])
    age = st.slider("Input age: ", 0, 120)
    hypertension = st.select_slider("Do you have hypertension: ", [0, 1])
    heart_disease = st.select_slider("Do you have heart disease: ", [0, 1])
    ever_married = st.select_slider("Have you ever married? ", [0, 1])
    avg_glucosis_lvl = st.slider("Average glucosis level: ", 50, 280)
    bmi = st.slider("Input Bmi: ", 10, 100)

    # User input data
    data = {
        "gender": gender,
        "work_type": work_type,
        "Residence_type": residence_status,
        "smoking_status": smoking_status,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "avg_glucose_level": avg_glucosis_lvl,
        "bmi": bmi
    }

    # Prediction button
    if st.button("Predict"):
        # Convert input data to a DataFrame
        X = pd.DataFrame([data])

        # Encode categorical features
        encoded_features = encoder.transform(X[categorical_features])

        # Get the feature names from the encoder
        feature_names = encoder.get_feature_names_out(input_features=categorical_features)

        # Create a DataFrame with the encoded features and feature names
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
        X_encoded = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)

        # Make predictions
        prediction_proba = lgb1.predict_proba(X_encoded)

        # Get SHAP values
        explainer = shap.TreeExplainer(lgb1)
        shap_values = explainer.shap_values(X_encoded)

        # Extract prediction probability and display it to the user
        probability = prediction_proba[0, 1]  # Assuming binary classification
        st.subheader(f"The predicted probability of stroke is {probability}.")
        st.subheader("IF you see result , higher than 0.3, we advice you to see a doctor")
        st.header("Shap forceplot")
        st.subheader("Features values impact on model made prediction")

        # Display SHAP force plot using Matplotlib
        shap.force_plot(explainer.expected_value[1], shap_values[1], features=X_encoded.iloc[0, :], matplotlib=True)

        # Save the figure to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=800)
        buf.seek(0)

        # Display the image in Streamlit
        st.image(buf, width=1100)

        # Display summary plot of feature importance
        shap.summary_plot(shap_values[1], X_encoded)

        # Display interaction summary plot
        shap_interaction_values = explainer.shap_interaction_values(X_encoded)
        shap.summary_plot(shap_interaction_values, X_encoded)

# Execute get_pred() only if the option is "Stroke prediction"
if option == "Stroke prediction":
    get_pred()

if option == "Model information":
    st.header("Light gradient boosting model")
    st.subheader("First tree of light gradient boosting model and how it makes decisions")
    st.image(r'lgbm_tree.png')

    st.subheader("Shap values visualization of how features contribute to model prediction")
    st.image(r'lgbm_model_shap_evaluation.png')