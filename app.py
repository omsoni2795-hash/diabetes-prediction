import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

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

# Sidebar - About & Explainable AI Info
st.sidebar.title("About")
st.sidebar.info(
    "This application predicts diabetes risk using Logistic Regression with Explainable AI."
)
st.sidebar.markdown("### 🔍 Model Explainability")
st.sidebar.markdown("""
- **SHAP Values**: Show how each feature impacts the prediction
- **Feature Importance**: Global model behavior
- **LIME**: Individual prediction explanations
""")

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

    # Display Risk Level
    col1, col2 = st.columns(2)
    with col1:
        if probability >= 0.7:
            st.error(f"🔴 High Risk ({probability*100:.2f}%)")
        elif probability >= 0.4:
            st.warning(f"🟡 Moderate Risk ({probability*100:.2f}%)")
        else:
            st.success(f"🟢 Low Risk ({probability*100:.2f}%)")

    with col2:
        st.metric("Diabetes Probability", f"{probability*100:.2f}%")

    # ==================== EXPLAINABLE AI SECTION ====================
    st.markdown("---")
    st.header("🔍 Explainable AI Analysis")

    # 1. FEATURE IMPORTANCE (Global)
    st.subheader("1. Global Feature Importance")
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance (Permutation-based)')
    st.pyplot(fig)

    # 2. SHAP Values (Local - Individual Prediction)
    st.subheader("2. SHAP Force Plot (Individual Prediction)")
    st.markdown("Shows how each feature pushes the prediction up (red) or down (blue) from the base value.")
    
    try:
        explainer = shap.LinearExplainer(model, X_train[:100])  # Use subset for faster computation
        shap_values = explainer.shap_values(user_scaled)
        
        # Create SHAP force plot
        fig_shap = shap.force_plot(
            explainer.expected_value, 
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            user_data,
            matplotlib=True
        )
        st.pyplot(fig_shap)
    except Exception as e:
        st.warning(f"SHAP visualization unavailable: {str(e)}")

    # 3. LIME Explanation (Local - Individual Prediction)
    st.subheader("3. LIME Local Explanation")
    st.markdown("Explains this specific prediction using locally interpretable model.")
    
    try:
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=X.columns,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification'
        )
        
        # Get explanation
        exp = lime_explainer.explain_instance(
            user_scaled[0],
            model.predict_proba,
            num_features=8
        )
        
        # Display LIME explanation
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
        
    except Exception as e:
        st.warning(f"LIME visualization unavailable: {str(e)}")

    # 4. Feature Contribution Summary
    st.subheader("4. Feature Contribution Summary")
    
    # Get coefficients from logistic regression
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("**Most Positive Impact (Increases Risk):**")
        positive = coefficients[coefficients['Coefficient'] > 0].head(3)
        for idx, row in positive.iterrows():
            st.write(f"• {row['Feature']}: {row['Coefficient']:.4f}")
    
    with col_right:
        st.write("**Most Negative Impact (Decreases Risk):**")
        negative = coefficients[coefficients['Coefficient'] < 0].head(3)
        for idx, row in negative.iterrows():
            st.write(f"• {row['Feature']}: {row['Coefficient']:.4f}")

    # 5. Input Value Context
    st.subheader("5. Your Input Values vs Training Data")
    
    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'Your Value': user_data.iloc[0].values,
        'Train Mean': X_train.mean(axis=0),
        'Train Std': X_train.std(axis=0)
    })

    st.dataframe(
        comparison_df.style.format(
            {
                'Your Value': '{:.2f}',
                'Train Mean': '{:.2f}',
                'Train Std': '{:.2f}',
            }
        )
    )
    
    st.markdown("---")