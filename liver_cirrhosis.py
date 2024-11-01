import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib


model = load_model("liver_cirrhosis_model.h5")

st.title("Liver Cirrhosis Stage Detection")
st.write("Predict the stage of liver cirrhosis based on patient data.")



def get_user_input():
    N_Days = st.number_input("N_Days", min_value=0)


    status_options = ['C (Censored)', 'CL (Censored due to liver tx)', 'D (Death)']
    Status = st.selectbox("Status", status_options)


    drug_options = ['D-penicillamine', 'Placebo']
    Drug = st.selectbox("Drug", drug_options)

    Age = st.number_input("Age (in days)", min_value=0)


    sex_options = ['Male', 'Female']
    Sex = st.selectbox("Sex", sex_options)


    ascites_options = ['Yes', 'No']
    Ascites = st.selectbox("Ascites", ascites_options)


    hepatomegaly_options = ['Yes', 'No']
    Hepatomegaly = st.selectbox("Hepatomegaly", hepatomegaly_options)


    spiders_options = ['Yes', 'No']
    Spiders = st.selectbox("Spiders", spiders_options)


    edema_options = ['No edema and no diuretic therapy for edema', 'Edema present without diuretics or edema resolved by diuretics', 'Edema despite diuretic therapy']
    Edema = st.selectbox("Edema", edema_options)

    Bilirubin = st.number_input("Bilirubin (mg/dl)", min_value=0.0)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0.0)
    Albumin = st.number_input("Albumin (gm/dl)", min_value=0.0)
    Copper = st.number_input("Copper (ug/day)", min_value=0.0)
    Alk_Phos = st.number_input("Alkaline Phosphatase (U/liter)", min_value=0.0)
    SGOT = st.number_input("SGOT (U/ml)", min_value=0.0)
    Tryglicerides = st.number_input("Triglycerides (mg/dl)", min_value=0.0)
    Platelets = st.number_input("Platelets (per ml/1000)", min_value=0)
    Prothrombin = st.number_input("Prothrombin time (s)", min_value=0.0)

    data = {
        'N_Days': N_Days,
        'Status': status_options.index(Status),
        'Drug': drug_options.index(Drug),
        'Age': Age,
        'Sex': sex_options.index(Sex),
        'Ascites': ascites_options.index(Ascites),
        'Hepatomegaly': hepatomegaly_options.index(Hepatomegaly),
        'Spiders': spiders_options.index(Spiders),
        'Edema': edema_options.index(Edema),
        'Bilirubin': Bilirubin,
        'Cholesterol': Cholesterol,
        'Albumin': Albumin,
        'Copper': Copper,
        'Alk_Phos': Alk_Phos,
        'SGOT': SGOT,
        'Tryglicerides': Tryglicerides,
        'Platelets': Platelets,
        'Prothrombin': Prothrombin
    }

    features = pd.DataFrame(data, index=[0])
    return features


scaler = joblib.load("scaler.joblib")
input_df = get_user_input()
scaled_input = scaler.transform(input_df)

if st.button("Predict Stage"):
    prediction = model.predict(scaled_input)
    predicted_stage = np.argmax(prediction)
    st.write(f"The predicted stage of liver cirrhosis is: {predicted_stage}")
