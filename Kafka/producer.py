from kafka import KafkaProducer
import json
import time
import numpy as np

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Sample transactions in the required format
transactions = [
    [-0.528593487, 3, -0.165851538, 1, 3, 3, 0.969748198, 0],
    [2.020675806, 1, -1.585569793, 2, 0, 3, -1.692208595, 0],
    [0.321691178, 0, -1.302209566, 0, 2, 4, -1.559110756, 0],
    [-0.083449461, 0, -0.730653886, 0, 0, 2, 0.17116116, 0],
    [-0.829229201, 3, -0.17332416, 0, 2, 2, 0.17116116, 0],
    [-0.829257875, 3, 1.467695596, 2, 2, 4, -0.827072638, 0],
    [-0.939378943, 1, -1.165544501, 2, 1, 3, -0.228132359, 0],
    [1.01847666, 2, -0.14035671, 0, 2, 3, 1.568688476, 0],
    [-0.077289525, 0, -0.748156605, 0, 3, 4, 1.102846037, 0],
    [0.235912363, 0, 1.430052762, 2, 0, 4, 0.637003599, 0]
]

# Define column names
columns = ["amount", "location", "time", "device", "transaction_type", "merchant_category", "customer_age", "is_fraud"]

# Sending Transactions
for txn in transactions:
    encoded_transaction = dict(zip(columns, txn))
    
    producer.send("test", encoded_transaction)
    print(f"✅ Sent: {encoded_transaction}")
    time.sleep(2)  # Simulate transaction delay

# Flush the Kafka producer
producer.flush()
print("🎯 All transactions sent successfully!")