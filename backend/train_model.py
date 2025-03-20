import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(file_path):
    """Trains a logistic regression model from a CSV file and returns accuracy results."""

    try:
        logging.info("✅ Loading dataset...")
        credit_card_data = pd.read_csv(file_path)

        # Remove temporary file after loading
        os.remove(file_path)

        logging.info("✅ File loaded successfully. Checking columns...")

        # Ensure the file contains 'Class' column
        if "Class" not in credit_card_data.columns:
            raise ValueError("CSV must contain a 'Class' column.")

        # Separate fraud (1) and legitimate (0) transactions
        legit = credit_card_data[credit_card_data.Class == 0]
        fraud = credit_card_data[credit_card_data.Class == 1]

        logging.info(f"🔹 Legit Transactions: {len(legit)}, Fraud Transactions: {len(fraud)}")

        # Ensure enough legitimate transactions for balancing
        if len(legit) < len(fraud):
            raise ValueError("Not enough legitimate transactions to balance the dataset.")

        # Balance the dataset
        legit_sample = legit.sample(n=len(fraud), random_state=42)
        new_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

        # Split features and target
        X = new_df.drop(columns='Class', axis=1)
        Y = new_df['Class']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        joblib.dump(scaler, "scaler.pkl")

        # Split into training & testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

        logging.info("🚀 Training Logistic Regression model...")

        # Train Logistic Regression model
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, Y_train)

        # Save the trained model
        joblib.dump(model, "logistic_model.pkl")

        logging.info("✅ Model trained and saved successfully.")

        # Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        test_accuracy = accuracy_score(Y_test, test_predictions)

        # Count fraud predictions
        fraud_detected_test = sum(test_predictions)
        total_fraud_present_test = sum(Y_test)

        logging.info(f"📊 Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"🕵️ Fraud Detected in Test Set: {fraud_detected_test}/{total_fraud_present_test}")

        return {
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "fraud_detected_test": fraud_detected_test,
            "total_fraud_present_test": total_fraud_present_test
        }

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise ValueError(f"Training failed: {str(e)}")
