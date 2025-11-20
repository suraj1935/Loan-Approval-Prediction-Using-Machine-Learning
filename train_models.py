import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

class RobustLoanDataPreprocessor:
    def __init__(self):
        # Separate base numeric columns from engineered numeric columns
        self.base_num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.num_cols = self.base_num_cols + ['TotalIncome', 'LoanAmountLog', 'EMI', 'BalanceIncome']
        self.cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                         'Credit_History', 'Property_Area']
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        # Fix data types for Dependents
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

        # Handle missing values only on base numeric columns first
        for col in self.base_num_cols:
            df[col] = df[col].fillna(df[col].median())

        # Missing values in categorical columns
        for col in self.cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Outlier treatment example (ApplicantIncome capped at 95th percentile)
        upper_limit = df['ApplicantIncome'].quantile(0.95)
        df.loc[df['ApplicantIncome'] > upper_limit, 'ApplicantIncome'] = upper_limit

        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

        # Encode categorical variables
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Scale all numeric columns including engineered ones
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])

        return df

    def transform(self, df):
        # Fix data types
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

        # Impute base numeric columns
        for col in self.base_num_cols:
            df[col] = df[col].fillna(df[col].median())

        # Impute categorical columns
        for col in self.cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

        # Encode categorical variables using saved encoders
        for col in self.cat_cols:
            le = self.label_encoders.get(col)
            if le:
                df[col] = le.transform(df[col].astype(str))
            else:
                # Fallback to fit transform if missing encoder, rare case
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Scale numeric columns
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        return df


def main():
    # Load training dataset (with target)
    df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

    # Inspect data
    print("Dataset shape:", df.shape)
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())

    # Map target: Y=1, N=0
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Separate features and target
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = df['Loan_Status']

    # Initialize and fit preprocessor
    preprocessor = RobustLoanDataPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled,
                                                      test_size=0.2, random_state=42)

    # Define models for ensemble
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)], voting='soft')

    # Train ensemble
    ensemble.fit(X_train, y_train)

    # Validation accuracy
    val_score = ensemble.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score:.4f}")

    # Save model and preprocessor
    joblib.dump(ensemble, 'ensemble_loan_model.pkl')
    joblib.dump(preprocessor, 'robust_preprocessor.pkl')
    print("Saved ensemble model and preprocessor as .pkl files.")


if __name__ == '__main__':
    main()
