import pandas as pd

# Define the dataset
data = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "V1": [-1.2, 1.5, -0.8, 2.1, -1.7, 0.9, -0.5, 1.2, -2.3, 1.4],
    "V2": [2.3, -0.4, 1.1, -1.5, 0.3, -2.2, 1.8, -1.3, 0.6, -0.7],
    "V3": [0.5, -1.2, -0.3, 0.7, 2.5, 1.1, -0.9, 2.1, -1.1, 0.3],
    "V4": [-0.8, 0.7, -2.1, 1.8, -1.1, -0.7, 2.3, -0.8, 1.5, -2.5],
    "V5": [1.5, -2.5, 0.5, -0.2, 1.8, 2.5, 0.6, 0.7, -0.9, 1.1],
    "V6": [0.3, 1.2, -0.9, -2.3, 0.7, -1.4, -1.1, -2.4, 2.3, -0.5],
    "Amount": [100.50, 200.00, 50.75, 300.10, 150.20, 250.30, 175.45, 220.60, 90.90, 275.80],
    "Class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Legit, 1 = Fraud
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = "creditcard_data.csv"
df.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully! ✅")
