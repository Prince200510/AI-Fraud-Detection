# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
df = pd.read_csv("strong_fraud_detection_dataset.csv")

# Handle missing values in the target column
df = df.dropna(subset=["is_fraud"])  # Drop rows with NaN target values

# Encode categorical features
label_encoder = LabelEncoder()
for col in ["location", "device", "transaction_type", "merchant_category"]:
    df[col] = label_encoder.fit_transform(df[col])

# Normalize numerical features
scaler = StandardScaler()
df[["amount", "time", "customer_age"]] = scaler.fit_transform(df[["amount", "time", "customer_age"]])

# Define features & target variable
X = df.drop("is_fraud", axis=1).values  
y = df["is_fraud"].astype(int).values  # Ensure target is integer type

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define a deep learning model for fraud detection
model = keras.Sequential([
    keras.layers.Dense(256, input_shape=(X.shape[1],), kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(32),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model with AdamW optimizer
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name="auc")]
)

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate model performance
loss, accuracy, auc_score = model.evaluate(X_test, y_test)
y_pred = (model.predict(X_test) > 0.5).astype(int)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"🎯 AUC-ROC Score: {roc_auc_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model
model.save("fraud_detection_model_v3.h5")
print("✅ High-Accuracy Model Saved as 'fraud_detection_model_v3.h5'")
