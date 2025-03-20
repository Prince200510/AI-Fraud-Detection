# import numpy as np
# import pandas as pd
# import joblib
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

# # Load Data
# def load_data(file_path):
#     """Loads CSV or Excel file into a DataFrame."""
#     if file_path.endswith(".csv"):
#         df = pd.read_csv(file_path)
#     elif file_path.endswith((".xls", ".xlsx")):
#         df = pd.read_excel(file_path)
#     else:
#         raise ValueError("Unsupported file format! Upload CSV or Excel.")

#     df.dropna(inplace=True)

#     if 'Class' not in df.columns:
#         raise ValueError("No 'Class' column found! Ensure dataset contains fraud labels.")

#     X = df.drop(columns=['Class'])  # Features
#     y = df['Class']  # Target
#     return X, y

# # Data balancing using SMOTE
# def balance_data(X, y):
#     """Balances fraud cases using SMOTE."""
#     smote = SMOTE(sampling_strategy=0.7, random_state=42)  # Increase fraud cases to 70% of normal cases
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     return X_resampled, y_resampled

# # Model training function
# def train_rf_model():
#     """Trains an optimized Random Forest model for fraud detection."""
#     file_path = input("📂 Enter file path (CSV/Excel): ")

#     try:
#         # Load data
#         X, y = load_data(file_path)
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Handle class imbalance
#         X_balanced, y_balanced = balance_data(X_scaled, y)
#         X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

#         # Define the model
#         model = RandomForestClassifier(random_state=42, class_weight="balanced_subsample")

#         # Define parameter grid
#         param_dist = {
#             "n_estimators": [200, 300, 400],  # More trees
#             "max_depth": [10, 20, None],  # Deeper trees
#             "min_samples_split": [2, 5, 10],  # Prevent overfitting
#             "min_samples_leaf": [1, 2, 4],  # Prevent too many splits
#             "bootstrap": [True, False],  # Try both bootstrapping options
#         }

#         # Perform Randomized Search for hyperparameter tuning
#         random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring="recall", n_jobs=-1, random_state=42)
#         random_search.fit(X_train, y_train)
#         best_model = random_search.best_estimator_

#         # Model evaluation
#         y_pred = best_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)

#         print(f"🔥 Model Accuracy: {accuracy * 100:.2f}%")
#         print(f"🔍 Fraud Detection Recall: {recall * 100:.2f}%")  # Focus on fraud recall
#         print("📊 Classification Report:\n", classification_report(y_test, y_pred))
#         print("🟢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#         # Save model and scaler
#         joblib.dump(best_model, "fraud_detection_rf.pkl")
#         joblib.dump(scaler, "scaler.pkl")
#         print("✅ Optimized model trained and saved successfully!")

#     except ValueError as e:
#         print(f"⚠️ {e}")

# if __name__ == "__main__":
#     train_rf_model()


# Deep learning 

# import numpy as np
# import pandas as pd
# import joblib
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import (
#     accuracy_score, recall_score, classification_report, confusion_matrix,
#     precision_recall_curve, roc_curve, auc, f1_score, matthews_corrcoef
# )

# # Configure Logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_data(file_path):
#     """Loads CSV/Excel file and returns features (X) and target (y)."""
#     try:
#         df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
#         df.dropna(inplace=True)
        
#         if 'Class' not in df.columns:
#             raise ValueError("No 'Class' column found! Ensure dataset contains fraud labels.")
        
#         X = df.drop(columns=['Class'])  
#         y = df['Class']  
        
#         logging.info(f"Dataset Loaded: {len(df)} transactions (Fraud: {sum(y == 1)}, Non-Fraud: {sum(y == 0)})")
#         return X, y
#     except Exception as e:
#         logging.error(f"Error loading data: {e}")
#         raise

# def build_nn_model(input_dim):
#     """Builds a deep neural network model for fraud detection."""
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=input_dim),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(32, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
        
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def train_nn_model():
#     """Trains an optimized TensorFlow Neural Network for fraud detection."""
#     file_path = input("📂 Enter file path (CSV/Excel): ")
    
#     try:
#         X, y = load_data(file_path)
        
#         # Splitting Data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#         logging.info(f"Data split: Train ({len(X_train)}), Test ({len(X_test)})")
        
#         # Feature Scaling
#         scaler = RobustScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         # Class Weights to Handle Imbalance
#         class_weight = {0: 1, 1: sum(y_train == 0) / sum(y_train == 1)}
        
#         # Build Model
#         model = build_nn_model(X_train.shape[1])
        
#         # Train Model
#         history = model.fit(
#             X_train_scaled, y_train, epochs=30, batch_size=64, 
#             validation_data=(X_test_scaled, y_test),
#             class_weight=class_weight, verbose=1
#         )
        
#         # Predictions
#         y_pred_proba = model.predict(X_test_scaled).flatten()
#         precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
#         fscore = np.nan_to_num((2 * precision * recall) / (precision + recall))
#         best_threshold = thresholds[np.argmax(fscore)] if len(thresholds) > 0 else 0.5  
#         y_pred = (y_pred_proba >= best_threshold).astype(int)
        
#         # Metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         mcc = matthews_corrcoef(y_test, y_pred)
#         auc_score = auc(*roc_curve(y_test, y_pred_proba)[:2])
        
#         logging.info(f"Model Accuracy: {accuracy * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1 * 100:.2f}% | MCC: {mcc:.2f} | AUC: {auc_score:.2f}")
        
#         # Classification Report
#         print("Classification Report:\n", classification_report(y_test, y_pred))
        
#         # Confusion Matrix
#         cm = confusion_matrix(y_test, y_pred)
#         plt.figure(figsize=(5, 4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()
        
#         # Save Model and Scaler
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         model_filename = f"nn_fraud_model_{timestamp}.h5"
#         scaler_filename = f"scaler_{timestamp}.pkl"
#         model.save(model_filename)
#         joblib.dump(scaler, scaler_filename)
#         logging.info(f"Model saved: {model_filename}, Scaler saved: {scaler_filename}")
        
#     except Exception as e:
#         logging.error(f"Error: {e}")

# if __name__ == "__main__":
#     train_nn_model()


# import numpy as np
# import pandas as pd
# import joblib
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from datetime import datetime
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import (
#     accuracy_score, recall_score, classification_report, confusion_matrix,
#     precision_recall_curve, roc_curve, auc, f1_score, matthews_corrcoef
# )

# # Configure Logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_data(file_path):
#     """Loads CSV/Excel file and returns features (X) and target (y)."""
#     try:
#         df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
#         df.dropna(inplace=True)
        
#         if 'Class' not in df.columns:
#             raise ValueError("No 'Class' column found! Ensure dataset contains fraud labels.")
        
#         X = df.drop(columns=['Class'])  
#         y = df['Class']  
        
#         logging.info(f"Dataset Loaded: {len(df)} transactions (Fraud: {sum(y == 1)}, Non-Fraud: {sum(y == 0)})")
#         return X, y
#     except Exception as e:
#         logging.error(f"Error loading data: {e}")
#         raise

# def build_nn_model(input_dim):
#     """Builds a deep neural network model for fraud detection."""
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=input_dim),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(32, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
        
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def train_and_evaluate():
#     """Trains and evaluates the model on the FULL dataset (no test split)."""
#     file_path = input("📂 Enter file path (CSV/Excel): ")
    
#     try:
#         X, y = load_data(file_path)
        
#         # Feature Scaling
#         scaler = RobustScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Class Weights to Handle Imbalance
#         class_weight = {0: 1, 1: sum(y == 0) / sum(y == 1)}
        
#         # Build Model
#         model = build_nn_model(X.shape[1])
        
#         # Train Model on 100% of Data
#         history = model.fit(
#             X_scaled, y, epochs=30, batch_size=64, 
#             class_weight=class_weight, verbose=1
#         )
        
#         # Predictions
#         y_pred_proba = model.predict(X_scaled).flatten()
#         precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
#         fscore = np.nan_to_num((2 * precision * recall) / (precision + recall))
#         best_threshold = thresholds[np.argmax(fscore)] if len(thresholds) > 0 else 0.5  
#         y_pred = (y_pred_proba >= best_threshold).astype(int)
        
#         # Metrics
#         accuracy = accuracy_score(y, y_pred)
#         recall = recall_score(y, y_pred)
#         f1 = f1_score(y, y_pred)
#         mcc = matthews_corrcoef(y, y_pred)
#         auc_score = auc(*roc_curve(y, y_pred_proba)[:2])
        
#         logging.info(f"Model Accuracy: {accuracy * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1 * 100:.2f}% | MCC: {mcc:.2f} | AUC: {auc_score:.2f}")
        
#         # Classification Report
#         print("Classification Report:\n", classification_report(y, y_pred))
        
#         # Confusion Matrix
#         cm = confusion_matrix(y, y_pred)
#         plt.figure(figsize=(5, 4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#         plt.show()
        
#         # Fraud Detection Count
#         fraud_detected = sum(y_pred[y == 1])
#         total_fraud = sum(y == 1)
#         logging.info(f"Fraud Detected: {fraud_detected} out of {total_fraud}")

#         # Save Model and Scaler
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         model_filename = f"nn_fraud_model_{timestamp}.h5"
#         scaler_filename = f"scaler_{timestamp}.pkl"
#         model.save(model_filename)
#         joblib.dump(scaler, scaler_filename)
#         logging.info(f"Model saved: {model_filename}, Scaler saved: {scaler_filename}")

#         return fraud_detected, total_fraud

#     except Exception as e:
#         logging.error(f"Error: {e}")

# if __name__ == "__main__":
#     detected, total = train_and_evaluate()
#     print(f"\n🚀 Detected {detected} out of {total} fraud cases!")


import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, recall_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc, f1_score, matthews_corrcoef
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads CSV/Excel file and returns features (X) and target (y)."""
    try:
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
        df.dropna(inplace=True)
        
        if 'Class' not in df.columns:
            raise ValueError("No 'Class' column found! Ensure dataset contains fraud labels.")
        
        X = df.drop(columns=['Class'])  
        y = df['Class']  
        
        logging.info(f"Dataset Loaded: {len(df)} transactions (Fraud: {sum(y == 1)}, Non-Fraud: {sum(y == 0)})")
        return X, y
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def build_nn_model(input_dim):
    """Builds a deep neural network model for fraud detection."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate():
    """Trains and evaluates the model on the FULL dataset (no test split)."""
    file_path = input("📂 Enter file path (CSV/Excel): ")
    
    try:
        X, y = load_data(file_path)
        
        # Feature Scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Class Weights to Handle Imbalance
        class_weight = {0: 1, 1: sum(y == 0) / sum(y == 1)}
        
        # Build Model
        model = build_nn_model(X.shape[1])
        
        # Train Model on 100% of Data
        history = model.fit(
            X_scaled, y, epochs=30, batch_size=64, 
            class_weight=class_weight, verbose=1
        )
        
        # Predictions
        y_pred_proba = model.predict(X_scaled).flatten()
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        fscore = np.nan_to_num((2 * precision * recall) / (precision + recall))
        best_threshold = thresholds[np.argmax(fscore)] if len(thresholds) > 0 else 0.5  
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        auc_score = auc(*roc_curve(y, y_pred_proba)[:2])
        
        logging.info(f"Model Accuracy: {accuracy * 100:.2f}% | Recall: {recall * 100:.2f}% | F1: {f1 * 100:.2f}% | MCC: {mcc:.2f} | AUC: {auc_score:.2f}")
        
        # Classification Report
        print("Classification Report:\n", classification_report(y, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        
        # Fraud Detection Count
        fraud_detected = sum(y_pred[y == 1])
        total_fraud = sum(y == 1)
        logging.info(f"Fraud Detected: {fraud_detected} out of {total_fraud}")

        # Save Model and Scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"nn_fraud_model_.pkl"
        scaler_filename = f"scaler_.pkl"
        
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        logging.info(f"Model saved: {model_filename}, Scaler saved: {scaler_filename}")

        return fraud_detected, total_fraud

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    detected, total = train_and_evaluate()
    print(f"\n🚀 Detected {detected} out of {total} fraud cases!")
