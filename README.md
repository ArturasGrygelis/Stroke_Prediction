Stroke Prediction Project
Overview
This project aims to develop a machine learning model to predict the likelihood of a stroke based on various health and demographic factors. The primary focus is on optimizing the recall metric to minimize false negatives, ensuring that individuals at high risk of stroke are not incorrectly classified as low-risk. The solution leverages a Light Gradient Boosting Machine (LightGBM) classifier, achieving a recall of 0.964 on test data.

Objective
The goal is to create a reliable predictive model using the Stroke Prediction Dataset from Kaggle. By prioritizing recall, the model is designed to support early intervention and reduce the risk of overlooking critical cases.

Key Features
Input Features: Age, BMI, average glucose levels, work type, smoking status, gender, hypertension, heart disease, marital status, and residence type.
Target Variable: Stroke occurrence (binary: 0 for no stroke, 1 for stroke).
Model: LightGBM Classifier, tuned for high recall.
Performance: Achieves a recall of 0.964 on the test dataset, indicating strong performance in identifying stroke cases.
Installation
To run this project locally, follow these steps:

Clone the Repository:
bash

Collapse

Wrap

Run

Copy
git clone <repository-url>
cd stroke-prediction
Install Dependencies: Ensure you have Python 3.11 or higher installed. Install the required packages using the provided requirements file:
bash

Collapse

Wrap

Run

Copy
pip install -r requirements.txt
Download the Dataset: Obtain the dataset from Kaggle and place it in the project directory as healthcare-dataset-stroke-data.csv.
Run the Application: Launch the Streamlit app to interact with the model:
bash

Collapse

Wrap

Run

Copy
streamlit run app.py
Usage
The project includes a Streamlit-based web interface (app.py) for stroke probability prediction and visualization.
Select the "Stroke Prediction" dashboard to input patient data (e.g., age, BMI, smoking status) and receive a probability score along with SHAP-based feature impact analysis.
The "Model Information" dashboard provides insights into the LightGBM model’s decision tree and feature importance.
Model Artifacts
Trained Model: Saved as lgb1_model.pkl using pickle.
Encoder: Saved as encoder.joblib for categorical feature transformation.
Feature Lists: Saved as categorical_features.joblib and features.joblib.
Results
The LightGBM model effectively predicts stroke risk, with the following key findings:

Recall: 0.964 on test data, ensuring high sensitivity to stroke cases.
Important Features: Age, BMI, average glucose levels, work type, and smoking status are the most influential predictors.
Potential Improvements
Data Collection: Gather additional data with more diverse or novel features to enhance model robustness.
Hyperparameter Tuning: Explore a broader range of hyperparameters to further optimize performance.
Feature Engineering: Experiment with models excluding certain features to assess their individual impact.
Contributing
Contributions to improve the model, documentation, or interface are welcome. Please fork the repository and submit a pull request with your changes. Ensure to follow the existing code style and include relevant tests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or feedback, please open an issue in the repository or contact the project maintainer.
