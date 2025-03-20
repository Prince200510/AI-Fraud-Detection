import os
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_logistic_model():
    """Creates and returns a Logistic Regression classifier."""
    return LogisticRegression(max_iter=500, random_state=42)

def train_logistic_model(file_path):
    """Loads data, trains the Logistic Regression model, and saves it."""
    try:
        logging.info("✅ Loading dataset...")
        data = pd.read_csv(file_path)

        # Remove temporary file after loading
        os.remove(file_path)

        logging.info("✅ File loaded successfully. Checking columns...")

        # Ensure the 'Class' column exists
        if "Class" not in data.columns:
            raise ValueError("CSV must contain a 'Class' column.")

        # Separate fraud (1) and legitimate (0) transactions
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

        logging.info(f"🔹 Legit Transactions: {len(legit)}, Fraud Transactions: {len(fraud)}")

        # Ensure dataset is balanced
        if len(legit) < len(fraud):
            raise ValueError("Not enough legitimate transactions to balance the dataset.")

        legit_sample = legit.sample(n=len(fraud), random_state=42)
        balanced_data = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

        # Prepare features & target
        X = balanced_data.drop(columns="Class", axis=1)
        y = balanced_data["Class"]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        joblib.dump(scaler, "scaler.pkl")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        logging.info("🚀 Training Logistic Regression model...")

        # Train Logistic Regression
        model = build_logistic_model()
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, "logistic_model_test3.pkl")
        logging.info("✅ Model trained and saved successfully.")

        # Evaluate Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
        logging.info("📊 Classification Report:\n" + classification_report(y_test, y_pred))
        logging.info("🟢 Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

        return {"accuracy": round(accuracy, 4)}

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise ValueError(f"Training failed: {str(e)}")

def load_logistic_model():
    """Loads and returns the trained Logistic Regression model."""
    try:
        return joblib.load("logistic_model_test3.pkl")
    except FileNotFoundError:
        logging.error("⚠️ Model file not found! Train the model first.")
        return None

if __name__ == "__main__":
    file_path = input("📂 Enter file path (CSV): ")
    train_logistic_model(file_path)
