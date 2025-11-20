import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Custom RobustLoanDataPreprocessor class (simplified but extend with full logic as needed)
class RobustLoanDataPreprocessor:
    def __init__(self):
        self.num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                         'TotalIncome', 'LoanAmountLog', 'EMI', 'BalanceIncome']
        self.cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                         'Credit_History', 'Property_Area']
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        # Fill missing values
        for col in self.num_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        for col in self.cat_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

        # Encode categorical variables
        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Scale numeric columns
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        return df

    def transform(self, df):
        # Fill missing values
        for col in self.num_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        for col in self.cat_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

        # Encode categorical variables with saved encoders
        for col in self.cat_cols:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = le.transform(df[col].astype(str))
                else:
                    # Fallback
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Scale numeric columns using saved scaler
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

# Load dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
    test_df = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
    return train_df, test_df

train_df, test_df = load_data()

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load('ensemble_loan_model.pkl')
    preprocessor = joblib.load('robust_preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

st.title("Loan Approval Prediction System")

# Sidebar shows dataset samples for reference
st.sidebar.header("Dataset Samples")
st.sidebar.write("Training Data Sample:")
st.sidebar.dataframe(train_df.head())
st.sidebar.write("Test Data Sample:")
st.sidebar.dataframe(test_df.head())

# User input form for prediction
with st.form("loan_input_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1)
    Loan_Amount_Term = st.number_input("Loan Amount Term (in months)", min_value=12)
    Credit_History = st.selectbox("Credit History", ["No", "Yes"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    input_dict = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': 1 if Credit_History == "Yes" else 0,
        'Property_Area': Property_Area
    }
    input_df = pd.DataFrame([input_dict])

    # Clean and preprocess user input
    processed_input = preprocessor.transform(input_df)

    # Predict loan approval
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0][1]

    result = "Approved" if prediction == 1 else "Rejected"
    st.subheader(f"Loan Prediction: {result}")
    st.write(f"Probability of Approval: {prediction_proba:.2f}")
