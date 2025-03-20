import joblib
import pandas as pd

def load_trained_model_Log():
    return joblib.load("logistic_test_2_model.pkl")

# Example usage
model = load_trained_model_Log()
print("Logistic Regression Model Loaded Successfully! ✅")
