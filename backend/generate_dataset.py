import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of transactions
num_samples = 2000

# Generate fake transaction data
data = {
    "Time": np.random.randint(0, 172, num_samples),  # Time in seconds (0 to 48 hours)
    "V1": np.random.randn(num_samples),
    "V2": np.random.randn(num_samples),
    "V3": np.random.randn(num_samples),
    "V4": np.random.randn(num_samples),
    "V5": np.random.randn(num_samples),
    "V6": np.random.randn(num_samples),
    "V7": np.random.randn(num_samples),
    "V8": np.random.randn(num_samples),
    "V9": np.random.randn(num_samples),
    "V10": np.random.randn(num_samples),
    "V11": np.random.randn(num_samples),
    "V12": np.random.randn(num_samples),
    "V13": np.random.randn(num_samples),
    "V14": np.random.randn(num_samples),
    "V15": np.random.randn(num_samples),
    "V16": np.random.randn(num_samples),
    "V17": np.random.randn(num_samples),
    "V18": np.random.randn(num_samples),
    "V19": np.random.randn(num_samples),
    "V20": np.random.randn(num_samples),
    "V21": np.random.randn(num_samples),
    "V22": np.random.randn(num_samples),
    "V23": np.random.randn(num_samples),
    "V24": np.random.randn(num_samples),
    "V25": np.random.randn(num_samples),
    "V26": np.random.randn(num_samples),
    "V27": np.random.randn(num_samples),
    "V28": np.random.randn(num_samples),
    "Amount": np.random.uniform(1, 5000, num_samples),  # Random transaction amounts
}

# Generate fraud labels (5% fraud transactions)
fraud_labels = np.zeros(num_samples)
fraud_indices = np.random.choice(num_samples, int(0.05 * num_samples), replace=False)
fraud_labels[fraud_indices] = 1

# Add fraud labels to dataset
data["Class"] = fraud_labels

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("creditcard_sample12345.csv", index=False)

print("✅ Sample dataset generated: creditcard_sample.csv")
