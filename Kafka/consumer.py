from kafka import KafkaConsumer
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime

# Load the trained model
model = tf.keras.models.load_model("fraud_detection_model_v3.h5")

# Load dataset to get location encoding
df = pd.read_csv("strong_fraud_detection_dataset.csv")

# Initialize Label Encoders for categorical fields
label_encoders = {}
categorical_columns = ['location', 'device', 'transaction_type', 'merchant_category']

for col in categorical_columns:
    if col in df.columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    else:
        print(f"⚠️ Warning: Column '{col}' not found in dataset.")

# Function to convert time (HH:MM AM/PM) to a numerical format
def convert_time_to_numeric(time_value):
    if isinstance(time_value, (int, float)):  # If already numeric, return as is
        return float(time_value)
    try:
        dt = datetime.strptime(time_value, "%I:%M %p")  # Convert to datetime object
        return dt.hour + dt.minute / 60.0  # Convert to a float value
    except (ValueError, TypeError):
        print(f"⚠️ Warning: Invalid time format '{time_value}', defaulting to 0")
        return 0  # Default value if time format is incorrect

# Kafka Consumer Configuration
consumer = KafkaConsumer(
    'test',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🔍 Waiting for transactions...")

for message in consumer:
    transaction = message.value
    print(f"📥 Received: {transaction}")

    # Extract transaction details with default values if missing
    amount = transaction.get("amount", 0)
    location = transaction.get("location", "Unknown")
    time_value = transaction.get("time", "12:00 AM")
    device = transaction.get("device", "Unknown")
    transaction_type = transaction.get("transaction_type", "Unknown")
    merchant_category = transaction.get("merchant_category", "Unknown")
    customer_age = transaction.get("customer_age", 0)

    # Convert time to numeric
    time_numeric = convert_time_to_numeric(time_value)

    # Encode categorical fields safely
    def encode_value(column, value):
        if column in label_encoders:
            try:
                return label_encoders[column].transform([value])[0]
            except ValueError:
                print(f"⚠️ Warning: Unknown {column} '{value}'. Assigning default code -1.")
                return -1  # Default encoding for unknown values
        return -1

    location_code = encode_value("location", location)
    device_code = encode_value("device", device)
    transaction_type_code = encode_value("transaction_type", transaction_type)
    merchant_category_code = encode_value("merchant_category", merchant_category)

    # Prepare input for model (convert NumPy types to Python int)
    input_data = np.array([[float(amount), location_code, time_numeric, device_code, transaction_type_code, merchant_category_code, float(customer_age)]], dtype=np.float32)

    # Predict fraud probability
    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        print(f"🚨 Fraud Alert! Transaction: {transaction} (Fraud Probability: {prediction:.2f})")
    else:
        print(f"✅ Safe Transaction: {transaction} (Fraud Probability: {prediction:.2f})")
