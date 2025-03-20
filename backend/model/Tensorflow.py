# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from preprocess import load_data, preprocess_data, balance_data

# def build_deep_model(input_shape):
#     """Builds and returns a deep learning model for fraud detection."""
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.2),
#         Dense(1, activation='sigmoid')
#     ])

#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def train_deep_model():
#     """Loads data, trains a deep learning model, and saves it."""
#     file_path = input("Enter file path (CSV/Excel): ")

#     try:
#         data = load_data(file_path)
#         X_scaled, y, scaler = preprocess_data(data)

#         if len(set(y)) < 2:
#             print("⚠️ Error: The dataset contains only one class! Cannot train.")
#             return

#         # Splitting the dataset
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#         # Handle class imbalance using class weights
#         class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#         class_weights = dict(enumerate(class_weights))

#         # Build and train the deep learning model
#         model = build_deep_model(X_train.shape[1])
#         model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights)

#         # Save model and scaler
#         model.save("fraud_detection_deep.h5")
#         joblib.dump(scaler, "scaler.pkl")

#         # Evaluate model accuracy
#         loss, accuracy = model.evaluate(X_test, y_test)
#         print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

#     except ValueError as e:
#         print(f"⚠️ {e}")

# def load_trained_deep_model():
#     """Loads and returns the trained deep learning fraud detection model."""
#     try:
#         return keras.models.load_model("fraud_detection_deep.h5")
#     except FileNotFoundError:
#         print("⚠️ Model file not found! Train the model first using `python model.py`.")
#         exit()

# if __name__ == "__main__":
#     train_deep_model()


import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import sys

def build_deep_model(input_shape):
    """Builds and returns a deep learning model for fraud detection."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data(file_path):
    """Loads CSV/Excel file and returns a DataFrame."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        return pd.read_excel(file_path)
    else:
        print("❌ Unsupported file format! Use CSV or Excel.")
        sys.exit(1)

def preprocess_data(data):
    """Preprocesses data: drops missing values, scales features."""
    data.dropna(inplace=True)

    if "Class" in data.columns:
        y = data["Class"].values
        X = data.drop(columns=["Class"]).values
    else:
        print("❌ Error: No 'Class' column found!")
        sys.exit(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_deep_model():
    """Trains and saves the deep learning fraud detection model."""
    file_path = input("📂 Enter file path (CSV/Excel): ")

    try:
        data = load_data(file_path)
        X_scaled, y, scaler = preprocess_data(data)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = dict(enumerate(class_weights))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = build_deep_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights)

        model.save("fraud_detection_deep.h5")
        joblib.dump(scaler, "scaler.pkl")

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"✅ Model trained successfully!\nAccuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"❌ Error: {e}")

def test_deep_model():
    """Loads trained model and tests it on a dataset."""
    try:
        model = load_trained_deep_model()
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("❌ Model or scaler not found! Train the model first.")
        sys.exit(1)

    file_path = input("📂 Enter test dataset file path (CSV/Excel): ")
    data = load_data(file_path)
    X_scaled, y, _ = preprocess_data(data)

    predictions = (model.predict(X_scaled).flatten() >= 0.5).astype(int)

    fraud_count = np.sum(predictions)
    actual_fraud = np.sum(y)
    print(f"\n🔍 Model detected {fraud_count} fraud cases (Actual: {actual_fraud})")

    print("\n📊 Classification Report:")
    print(classification_report(y, predictions, target_names=["Legit", "Fraud"]))

    print("\n🟢 Confusion Matrix:\n", confusion_matrix(y, predictions))

def load_trained_deep_model():
    """Loads the trained deep learning fraud detection model."""
    try:
        return keras.models.load_model("fraud_detection_deep.h5")
    except FileNotFoundError:
        print("❌ Model file not found! Train the model first.")
        sys.exit(1)

if __name__ == "__main__":
    print("\nChoose an option:\n1️⃣ Train Model\n2️⃣ Test Model")
    choice = input("👉 Enter (1/2): ")

    if choice == "1":
        train_deep_model()
    elif choice == "2":
        test_deep_model()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
