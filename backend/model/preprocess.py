import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Loads dataset from a CSV or Excel file."""
    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Upload a CSV or Excel file.")
        return data
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def preprocess_data(data):
    """Handles missing values and scales features."""
    data.dropna(inplace=True)

    if "Class" not in data.columns:
        raise ValueError("Error: The dataset must contain a 'Class' column.")

    X = data.drop(columns=['Class'])
    y = data['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def balance_data(X, y):
    """Handles class imbalance using SMOTE."""
    if len(set(y)) < 2:
        raise ValueError("Error: Dataset contains only one class. Cannot balance data.")

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled
