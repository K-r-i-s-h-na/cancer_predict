import pickle
import streamlit as st
import numpy as np

# Load trained model
w = pickle.load(open(r'C:\internship\cancer prediction\lungs.sav', 'rb'))

def predict(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = w.predict(input_array)
    return 'Survived' if prediction[0] == 1 else 'Not Survived'

def main():
    st.title("🧬 Cancer Survival Prediction App")

    # Inputs
    age = st.number_input("Age", min_value=0.0)
    gender = st.selectbox("Gender", ['Male', 'Female'])  # Male=1, Female=0
    cancer_stage = st.selectbox("Cancer Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])  # Manual encoding
    family_history = st.selectbox("Family History of Cancer", [0, 1])
    smoking_status = st.selectbox("Smoking Status (0=Never, 1=Former, 2=Current, 3=Heavy)", [0, 1, 2, 3])
    bmi = st.number_input("BMI", min_value=0.0)
    cholesterol_level = st.number_input("Cholesterol Level", min_value=0.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    asthma = st.selectbox("Asthma", [0, 1])
    cirrhosis = st.selectbox("Cirrhosis", [0, 1])
    other_cancer = st.selectbox("Other Cancer", [0, 1])
    treatment_type = st.selectbox("Treatment Type", [0, 1, 2, 3])  # already numeric
    treatment_duration = st.number_input("Treatment Duration (Days)", min_value=0.0)

    # Encoding
    gender_map = {'Male': 1, 'Female': 0}
    stage_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}

    if st.button("🔍 Predict Survival"):
        try:
            input_list = [
                age,
                gender_map[gender],
                stage_map[cancer_stage],
                family_history,
                smoking_status,
                bmi,
                cholesterol_level,
                hypertension,
                asthma,
                cirrhosis,
                other_cancer,
                treatment_type,
                treatment_duration
            ]
            result = predict(input_list)
            st.success(f"✅ Predicted Outcome: {result}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
