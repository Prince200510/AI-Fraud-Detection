import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a Sample Dataset (You can replace this with your own dataset)
np.random.seed(42)

# Generate random values for Time, V1-V28, Amount, and Class
n_samples = 1000  # Change this to the number of rows you want
time = np.random.randint(1, 100000, size=n_samples)
amount = np.random.uniform(1, 5000, size=n_samples)
class_label = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # 95% non-fraud, 5% fraud

# Create V1 to V28 with random values
features = np.random.randn(n_samples, 28)  # Replace with real data if available

# Combine into a DataFrame
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
df = pd.DataFrame(np.column_stack([time, features, amount, class_label]), columns=columns)

print("Original Dataset:\n", df.head())

# Step 2: Apply PCA for Feature Reduction
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df.iloc[:, 1:29])  # Standardize V1 to V28

pca = PCA(n_components=10)  # Choose number of components
pca_features = pca.fit_transform(features_scaled)

# Create a new DataFrame with PCA features
pca_columns = [f'PC{i+1}' for i in range(pca.n_components_)]
df_pca = pd.DataFrame(np.column_stack([df['Time'], pca_features, df['Amount'], df['Class']]), 
                       columns=['Time'] + pca_columns + ['Amount', 'Class'])

print("\nPCA Transformed Dataset:\n", df_pca.head())

# Save to CSV (optional)
df.to_csv("original_dataset.csv", index=False)
df_pca.to_csv("pca_transformed_dataset.csv", index=False)
