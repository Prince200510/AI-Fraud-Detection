import numpy as np
import joblib
import pandas as pd
from model import load_trained_model
from preprocess import load_data, preprocess_data

def detect_fraud(model, scaler, data):
    """Predicts fraud in the dataset."""
    X, y, _ = preprocess_data(data)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    fraud_indices = np.where(predictions == 1)[0]
    fraud_transactions = data.iloc[fraud_indices]
    fraud_count = len(fraud_transactions)

    return fraud_count, fraud_transactions

def main():
    """Main function to load model, process data, and detect fraud."""
    print("Loading trained model...")
    model = load_trained_model()
    scaler = joblib.load("scaler.pkl")

    file_path = input("Enter file path (CSV/Excel): ")
    print("Loading transaction data...")
    data = load_data(file_path)

    print("Detecting fraudulent transactions...")
    fraud_count, fraud_transactions = detect_fraud(model, scaler, data)

    print(f"Total Fraudulent Transactions Detected: {fraud_count}")
    if fraud_count > 0:
        print("Fraudulent Transactions:")
        print(fraud_transactions[['Time', 'Amount', 'Class']])
    else:
        print("No fraudulent transactions detected.")

if __name__ == "__main__":
    main()
