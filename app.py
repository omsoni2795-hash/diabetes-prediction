import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# Frontend
st.title("🩺 AI Diabetes Risk Predictor")
st.markdown("Enter your health details below.")
# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This application predicts diabetes risk using Logistic Regression."
)

st.write(f"Model Accuracy: {accuracy*100:.2f}%")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    user_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })


    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    if probability >= 0.7:
        st.error(f"High Risk ({probability*100:.2f}%)")
    elif probability >= 0.4:
        st.warning(f"Moderate Risk ({probability*100:.2f}%)")
    else:
        st.success(f"Low Risk ({probability*100:.2f}%)")

    st.write(f"Probability of Diabetes: {probability*100:.2f}%")