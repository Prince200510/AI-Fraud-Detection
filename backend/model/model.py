# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from preprocess import load_data, preprocess_data, balance_data

# def build_model():
#     """Creates and returns a Random Forest classifier."""
#     return RandomForestClassifier(n_estimators=100, random_state=42)

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

#         X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#         model = build_model()
#         model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
#         print("📊 Classification Report:\n", classification_report(y_test, y_pred))
        
#         joblib.dump(model, "fraud_detection_model.pkl")
#         joblib.dump(scaler, "scaler.pkl")
#         print("✅ Model trained and saved successfully!")

#     except ValueError as e:
#         print(f"⚠️ {e}")

# def load_trained_model():
#     """Loads and returns the trained fraud detection model."""
#     try:
#         return joblib.load("fraud_detection_model.pkl")
#     except FileNotFoundError:
#         print("⚠️ Model file not found! Train the model first using `python model.py`.")
#         exit()

# if __name__ == "__main__":
#     train_model()

# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from preprocess import load_data, preprocess_data, balance_data

# def build_rf_model():
#     """Creates and returns a Random Forest classifier."""
#     return RandomForestClassifier(n_estimators=100, random_state=42)

# def train_rf_model():
#     """Loads data, trains the Random Forest model, and saves it."""
#     file_path = input("📂 Enter file path (CSV/Excel): ")
    
#     try:
#         data = load_data(file_path)
#         X_scaled, y, scaler = preprocess_data(data)

#         if len(set(y)) < 2:
#             print("⚠️ Error: The dataset contains only one class! Cannot train.")
#             return
        
#         X_resampled, y_resampled = balance_data(X_scaled, y)
#         X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
#         model = build_rf_model()
#         model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
#         print("📊 Classification Report:\n", classification_report(y_test, y_pred))
#         print("🟢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
#         joblib.dump(model, "fraud_detection_rf.pkl")
#         joblib.dump(scaler, "scaler.pkl")
#         print("✅ Model trained and saved successfully!")
    
#     except ValueError as e:
#         print(f"⚠️ {e}")
        
# def evaluate_models(X_test, y_test):
#     """Evaluate the Random Forest model and return classification reports & confusion matrices."""
#     rf_model = load_trained_model()  # Load only the RF model

#     if not rf_model:
#         return {"error": "Random Forest model could not be loaded!"}

#     # Predictions
#     rf_predictions = rf_model.predict(X_test)

#     # Classification Report
#     rf_report = classification_report(y_test, rf_predictions, output_dict=True)

#     # Confusion Matrix
#     rf_conf_matrix = confusion_matrix(y_test, rf_predictions).tolist()

#     return {
#         "rf_report": rf_report,
#         "rf_conf_matrix": rf_conf_matrix
#     }


# def load_trained_model():
#     """Loads and returns the trained Random Forest fraud detection model."""
#     try:
#         return joblib.load("fraud_detection_model.pkl")
#     except FileNotFoundError:
#         print("⚠️ Model file not found! Train the model first using `python rf_model.py`.")
#         exit()

# if __name__ == "__main__":
#     train_rf_model()


import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

# Load and preprocess data
def load_data(file_path):
    """Loads CSV or Excel file into a DataFrame."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format! Upload CSV or Excel.")

    df.dropna(inplace=True)

    if 'Class' not in df.columns:
        raise ValueError("No 'Class' column found! Ensure dataset contains fraud labels.")

    X = df.drop(columns=['Class'])  # Features
    y = df['Class']  # Target
    return X, y

# Data balancing using SMOTE
def balance_data(X, y):
    """Balances fraud cases using SMOTE."""
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Fraud cases = 50% of non-fraud
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Model training function
def train_rf_model():
    """Trains Random Forest model with class balancing & hyperparameter tuning."""
    file_path = input("📂 Enter file path (CSV/Excel): ")
    
    try:
        # Load data
        X, y = load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance
        X_balanced, y_balanced = balance_data(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        model = RandomForestClassifier(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": [{0:1, 1:10}, {0:1, 1:15}]
        }

        grid_search = GridSearchCV(model, params, cv=3, scoring="recall", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Model evaluation
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
        print(f"🔍 Fraud Detection Recall: {recall * 100:.2f}%")  # Focus on fraud recall
        print("📊 Classification Report:\n", classification_report(y_test, y_pred))
        print("🟢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Save model and scaler
        joblib.dump(best_model, "fraud_detection_rf.pkl")
        joblib.dump(scaler, "scaler.pkl")
        print("✅ Model trained and saved successfully!")

    except ValueError as e:
        print(f"⚠️ {e}")

# Load trained model
def load_trained_model():
    """Loads the trained Random Forest model."""
    try:
        return joblib.load("fraud_detection_rf.pkl")
    except FileNotFoundError:
        print("⚠️ Model file not found! Train the model first using `python fraud_model.py`.")
        exit()

# Evaluate model performance
def evaluate_model(X_test, y_test):
    """Evaluate the model and return classification reports & confusion matrix."""
    rf_model = load_trained_model()
    if not rf_model:
        return {"error": "Model could not be loaded!"}

    rf_predictions = rf_model.predict(X_test)
    rf_report = classification_report(y_test, rf_predictions, output_dict=True)
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions).tolist()

    return {"rf_report": rf_report, "rf_conf_matrix": rf_conf_matrix}


if __name__ == "__main__":
    train_rf_model()
