# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from preprocess import load_data, preprocess_data, balance_data

# def build_model():
#     """Creates and returns a Logistic Regression model."""
#     return LogisticRegression(solver='liblinear', random_state=42)

# def train_model():
#     """Loads data, trains the model, and saves it."""
#     file_path = input("Enter file path (CSV/Excel): ")

#     try:
#         data = load_data(file_path)
#         X_scaled, y, scaler = preprocess_data(data)

#         if len(set(y)) < 2:
#             print("⚠️ Error: The dataset contains only one class! Cannot train.")
#             return

#         X_resampled, y_resampled = balance_data(X_scaled, y)
        
#         # Split dataset
#         X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
#         model = build_model()
#         model.fit(X_train, y_train)
        
#         # Save model and scaler
#         joblib.dump(model, "fraud_detection_logreg.pkl")
#         joblib.dump(scaler, "scaler.pkl")
        
#         # Model accuracy
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred) * 100
#         print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")

#     except ValueError as e:
#         print(f"⚠️ {e}")

# def load_trained_model_Log():
#     """Loads and returns the trained Logistic Regression fraud detection model."""
#     try:
#         return joblib.load("fraud_detection_logreg.pkl")
#     except FileNotFoundError:
#         print("⚠️ Model file not found! Train the model first using `python model.py`.")
#         exit()

# if __name__ == "__main__":
#     train_model()

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from preprocess import load_data, preprocess_data, balance_data
from imblearn.over_sampling import SMOTE

def build_model():
    """Creates and returns a Logistic Regression model with hyperparameter tuning."""
    return LogisticRegression(solver='liblinear', random_state=42, C=1.0)

def train_model():
    """Loads data, trains the model, and saves it."""
    file_path = input("Enter file path (CSV/Excel): ")

    try:
        data = load_data(file_path)
        X_scaled, y, scaler = preprocess_data(data)

        if len(set(y)) < 2:
            print("⚠️ Error: The dataset contains only one class! Cannot train.")
            return

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        model = build_model()
        model.fit(X_train, y_train)

        # Save model and scaler
        joblib.dump(model, "fraud_detection_logreg.pkl")
        joblib.dump(scaler, "scaler.pkl")

        # Model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")

        # Additional Evaluation Metrics
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nAUC-ROC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    except ValueError as e:
        print(f"⚠️ {e}")

def load_trained_model_Log():
    """Loads and returns the trained Logistic Regression fraud detection model."""
    try:
        return joblib.load("fraud_detection_logreg.pkl")
    except FileNotFoundError:
        print("⚠️ Model file not found! Train the model first using `python model.py`.")
        exit()

if __name__ == "__main__":
    train_model()
