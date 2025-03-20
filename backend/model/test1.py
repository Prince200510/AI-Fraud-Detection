import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load and preprocess data
def load_data(file_path):
    """Loads and preprocesses the dataset from CSV, handling NaN values."""
    data = pd.read_csv(file_path)
    
    # Check for missing values in the dataset
    if data.isnull().values.any():
        print("Warning: The dataset contains missing values. Cleaning the dataset...")
        
        # Option 1: Drop rows with NaN values
        data = data.dropna()
        
        # Option 2: Alternatively, you could fill missing values with the mean or median
        # data.fillna(data.mean(), inplace=True)  # Replace NaNs with the column mean (for numerical columns)
        # data.fillna(data.mode().iloc[0], inplace=True)  # Replace NaNs with the column mode (for categorical columns)
    
    return data

def preprocess_data(data):
    """Preprocess data: Splitting and scaling features."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTEENN balancing technique
    smote_enn = SMOTEENN()
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler

# Function to train the Random Forest model and Neural Network
def train_models(file_path):
    """Trains both Random Forest and Neural Network models and saves them."""
    data = load_data(file_path)
    X_train_resampled, X_test, y_train_resampled, y_test, scaler = preprocess_data(data)

    # Print class distribution in the original dataset
    print("Class distribution in the original dataset:")
    print(data['Class'].value_counts())
    
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    
    # Feature importance for Random Forest (optional)
    feature_importances = rf.feature_importances_
    selected_features = np.argsort(feature_importances)[-30:]  # Select top 30 features (same as training features)
    X_train_selected = X_train_resampled[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Neural Network (TensorFlow model)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_selected, y_train_resampled, epochs=20, batch_size=32, validation_data=(X_test_selected, y_test))

    # Save both models and the scaler
    joblib.dump(rf, "fraud_detection_rf.pkl")
    model.save("fraud_detection_nn.h5")
    joblib.dump(scaler, "scaler.pkl")

    return rf, model, scaler, X_test_selected, y_test

# Function to load trained models
def load_trained_models():
    """Loads the trained Random Forest and Neural Network models."""
    try:
        rf_model = joblib.load("fraud_detection_rf.pkl")
        nn_model = tf.keras.models.load_model("fraud_detection_nn.h5")
        scaler = joblib.load("scaler.pkl")
        return rf_model, nn_model, scaler
    except FileNotFoundError:
        print("⚠️ Model files not found! Train the model first using `python model.py`.")
        exit()

# Function to make predictions using both models
def predict_fraud(input_data):
    """Predicts fraud based on input data using the trained models."""
    rf_model, nn_model, scaler = load_trained_models()

    # Preprocess and scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Use the same number of features (30) for prediction
    feature_importances = rf_model.feature_importances_
    selected_features = np.argsort(feature_importances)[-30:]
    input_data_selected = input_data_scaled[:, selected_features]

    # Predictions using both models
    rf_predictions = rf_model.predict(input_data_selected)
    nn_predictions = (nn_model.predict(input_data_selected) > 0.5).astype("int32")

    return rf_predictions, nn_predictions

# Function to evaluate models
# Function to evaluate models
def evaluate_models_test1(X_test_selected, y_test):
    """Evaluates the Random Forest and Neural Network models."""
    rf_model, nn_model, scaler = load_trained_models()

    # Random Forest prediction
    rf_predictions = rf_model.predict(X_test_selected)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_precision = precision_score(y_test, rf_predictions)
    rf_recall = recall_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)

    # Neural Network prediction
    nn_predictions = (nn_model.predict(X_test_selected) > 0.5).astype("int32")
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    nn_precision = precision_score(y_test, nn_predictions)
    nn_recall = recall_score(y_test, nn_predictions)
    nn_f1 = f1_score(y_test, nn_predictions)

    print("Random Forest Evaluation:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall: {rf_recall:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")

    print("\nNeural Network Evaluation:")
    print(f"Accuracy: {nn_accuracy:.4f}")
    print(f"Precision: {nn_precision:.4f}")
    print(f"Recall: {nn_recall:.4f}")
    print(f"F1 Score: {nn_f1:.4f}")
    
    # Actual number of frauds in the test set
    total_actual_frauds = np.sum(y_test == 1)
    print(f"\nTotal actual fraud instances in the test set: {total_actual_frauds}")

    # Predicted frauds by Random Forest
    total_rf_frauds = np.sum(rf_predictions == 1)
    print(f"Predicted frauds by Random Forest: {total_rf_frauds}")

    # Predicted frauds by Neural Network
    total_nn_frauds = np.sum(nn_predictions == 1)
    print(f"Predicted frauds by Neural Network: {total_nn_frauds}")
    
    # Class distribution of predicted values
    print("\nPredicted class distribution (Random Forest):")
    print(pd.Series(rf_predictions).value_counts())

    print("\nPredicted class distribution (Neural Network):")
    print(pd.Series(nn_predictions.flatten()).value_counts())
    
    return rf_predictions, nn_predictions


if __name__ == "__main__":
    # Path to dataset (ensure to change this according to your local path)
    file_path = "creditcard.csv"
    
    # Train and evaluate models
    rf_model, nn_model, scaler, X_test_selected, y_test = train_models(file_path)
    rf_predictions, nn_predictions = evaluate_models_test1(X_test_selected, y_test)

    # Optionally, print predicted class for each instance in the test set
    print("\nPredicted classes for each test instance:")
    for i, (rf_pred, nn_pred) in enumerate(zip(rf_predictions, nn_predictions)):
        print(f"Instance {i+1}: Random Forest Prediction: {rf_pred}, Neural Network Prediction: {nn_pred}")
