# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load dataset
# credit_card_data = pd.read_csv('creditcard.csv')

# # Separate fraud and legitimate transactions
# legit = credit_card_data[credit_card_data.Class == 0]
# fraud = credit_card_data[credit_card_data.Class == 1]

# # Sample equal number of legitimate transactions as fraud
# legit_sample = legit.sample(n=len(fraud), random_state=42)

# # Create a new balanced dataset
# new_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

# # Split features and target
# X = new_df.drop(columns='Class', axis=1)
# Y = new_df['Class']

# # Split data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Train Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # Save the trained model
# joblib.dump(model, "logistic_test_2_model.pkl")
# print("Model saved as logistic_test_2_model.pkl ✅")

# # Predict and calculate accuracy
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# # Count fraud transactions detected
# fraud_detected = sum(X_test_prediction)
# total_fraud_present = sum(Y_test)

# # Print results
# print('Accuracy on Training Data:', training_data_accuracy)
# print('Accuracy on Test Data:', test_data_accuracy)
# print('Total Fraud Transactions Detected:', fraud_detected)
# print('Total Fraud Transactions Present in Test Data:', total_fraud_present)


# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load dataset
# credit_card_data = pd.read_csv('creditcard1.csv')

# # Separate fraud and legitimate transactions
# legit = credit_card_data[credit_card_data.Class == 0]
# fraud = credit_card_data[credit_card_data.Class == 1]

# # Sample equal number of legitimate transactions as fraud
# legit_sample = legit.sample(n=len(fraud), random_state=42)

# # Create a new balanced dataset
# new_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

# # Split features and target
# X = new_df.drop(columns='Class', axis=1)
# Y = new_df['Class']

# # Split data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Train Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # Save the trained model
# joblib.dump(model, "logistic_test_2_model.pkl")
# print("Model saved as logistic_test_2_model.pkl ✅")

# # Predict and calculate accuracy
# X_train_prediction = model.predict(X_train)
# X_test_prediction = model.predict(X_test)

# # Count fraud transactions detected in test set
# fraud_detected_test = sum(X_test_prediction)
# total_fraud_present_test = sum(Y_test)

# # Count fraud transactions detected in total dataset (if applicable)
# X_full_prediction = model.predict(X)  # Predict on the full dataset
# fraud_detected_total = sum(X_full_prediction)

# # Print debugging information
# print("\n🔍 Debugging Information:")
# print(f"🔹 Test Set Fraud Predictions: {fraud_detected_test}")
# print(f"🔹 Actual Fraud in Test Set: {total_fraud_present_test}")
# print(f"🔹 Total Fraud Predicted (Full Dataset): {fraud_detected_total}")  # Should match your UI


# def load__log_trained_model():
#     """Loads and returns the trained Random Forest fraud detection model."""
#     try:
#         return joblib.load("logistic_test_2_model.pkl")
#     except FileNotFoundError:
#         print("⚠️ Model file not found! Train the model first using `python rf_model.py`.")
#         exit()
        
# # Return values (for API or further usage)
# result = {
#     "test_fraud_detected": fraud_detected_test,
#     "total_fraud_present_test": total_fraud_present_test,
#     "total_fraud_predicted": fraud_detected_total
# }



import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset from command-line argument
if len(sys.argv) < 2:
    print("Error: No file provided.")
    sys.exit(1)

file_path = sys.argv[1]
credit_card_data = pd.read_csv(file_path)

# Separate fraud and legitimate transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Sample equal number of legitimate transactions as fraud
legit_sample = legit.sample(n=len(fraud), random_state=42)

# Create a balanced dataset
new_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

# Split features and target
X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Save the trained model
joblib.dump(model, "logistic_model.pkl")

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

# Count fraud predictions
fraud_detected_test = sum(test_predictions)
total_fraud_present_test = sum(Y_test)

# Print debugging information (this will be returned to the frontend)
print("\n🔍 Debugging Information:")
print(f"🔹 Training Accuracy: {train_accuracy:.4f}")
print(f"🔹 Test Accuracy: {test_accuracy:.4f}")
print(f"🔹 Fraud Predictions in Test Set: {fraud_detected_test}/{total_fraud_present_test}")
