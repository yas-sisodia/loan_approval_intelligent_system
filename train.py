import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def train_and_export():
    df = pd.read_csv("loan_data.csv")
    df_clean = df.drop(columns=["Loan_ID"]).copy()
    df_clean["Loan_Status"] = df_clean["Loan_Status"].map({"Y": 1, "N": 0})
    
    X = df_clean.drop(columns=["Loan_Status"])
    y = df_clean["Loan_Status"].astype(int)
    
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, "best_loan_model_pipeline.joblib")
    print("Model trained and serialized successfully in cloud environment.")

if __name__ == "__main__":
    train_and_export()