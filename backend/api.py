# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import pandas as pd
# import joblib
# import io
# from model.model import load_trained_model
# from model.LogisticRegression import load_trained_model_Log
# # from model.Tensorflow import load_trained_deep_model
# from model.test1 import load_trained_models
# from preprocess import preprocess_data

# # Load both trained models
# rf_model = load_trained_model()  # Random Forest model
# lr_model = load_trained_model_Log()  # Logistic Regression model
# dp_model = load_trained_models()
# scaler = joblib.load("scaler.pkl")

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Adjust if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def home():
#     return {"message": "Welcome to the Fraud Detection API!"}

# @app.post("/predict/rf/")
# async def predict_rf(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using the Random Forest model."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Random Forest
#         X_scaled = scaler.transform(df)
#         rf_predictions = rf_model.predict(X_scaled)

#         # Calculate fraud detection counts
#         rf_fraud_count = np.sum(rf_predictions)

#         return {"rf_fraud_count": int(rf_fraud_count)}

#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/predict/lr/")
# async def predict_lr(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using the Logistic Regression model."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Logistic Regression
#         X_scaled = scaler.transform(df)
#         lr_predictions = lr_model.predict(X_scaled)

#         # Calculate fraud detection counts
#         lr_fraud_count = np.sum(lr_predictions)

#         return {"lr_fraud_count": int(lr_fraud_count)}

#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/predict/dp/")
# async def predict_dp(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using the Deep Learning (TensorFlow) model."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Deep Learning (TensorFlow)
#         X_scaled = scaler.transform(df)
#         dp_predictions = (dp_model.predict(X_scaled).flatten() >= 0.5).astype(int)

#         # Calculate fraud detection counts
#         dp_fraud_count = np.sum(dp_predictions)

#         return {"dp_fraud_count": int(dp_fraud_count)}

#     except Exception as e:
#         return {"error": str(e)}


# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import pandas as pd
# import joblib
# import io
# from model.model import load_trained_model
# from model.LogisticRegression import load_trained_model_Log
# # from model.Tensorflow import load_trained_deep_model
# from model.test1 import load_trained_models
# from preprocess import preprocess_data

# # Load both trained models
# rf_model = load_trained_model()  # Random Forest model
# lr_model = load_trained_model_Log()  # Logistic Regression model
# dp_model = load_trained_models()  # Deep Learning model (TensorFlow)
# scaler = joblib.load("scaler.pkl")

# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Adjust if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def home():
#     return {"message": "Welcome to the Fraud Detection API!"}

# @app.post("/predict/rf/")
# async def predict_rf(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using the Random Forest model."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Random Forest
#         X_scaled = scaler.transform(df)
#         rf_predictions = rf_model.predict(X_scaled)

#         # Calculate fraud detection counts
#         rf_fraud_count = np.sum(rf_predictions)

#         return {"rf_fraud_count": int(rf_fraud_count), "rf_predictions": rf_predictions.tolist()}

#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/predict/lr/")
# async def predict_lr(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using the Logistic Regression model."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Logistic Regression
#         X_scaled = scaler.transform(df)
#         lr_predictions = lr_model.predict(X_scaled)

#         # Calculate fraud detection counts
#         lr_fraud_count = np.sum(lr_predictions)

#         return {"lr_fraud_count": int(lr_fraud_count), "lr_predictions": lr_predictions.tolist()}

#     except Exception as e:
#         return {"error": str(e)}
# @app.post("/predict/dp/")
# async def predict(file: UploadFile = File(...)):
#     """Processes an uploaded file and predicts fraud using both Random Forest and Neural Network models."""
#     try:
#         content = await file.read()
#         file_ext = file.filename.split(".")[-1]

#         # Read uploaded file
#         if file_ext == "csv":
#             df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         elif file_ext in ["xls", "xlsx"]:
#             df = pd.read_excel(io.BytesIO(content))
#         else:
#             return {"error": "Unsupported file format! Upload a CSV or Excel file."}

#         df.dropna(inplace=True)

#         # Save the "Class" column if it exists
#         class_column = df["Class"].copy() if "Class" in df.columns else None
#         if "Class" in df.columns:
#             df.drop(columns=["Class"], inplace=True)

#         # Preprocess and predict using Random Forest and Neural Network
#         X_scaled = scaler.transform(df)

#         # Random Forest Prediction
#         rf_predictions = rf_model.predict(X_scaled)
#         rf_fraud_count = np.sum(rf_predictions)  # Counting number of frauds predicted by RF

#         # Neural Network Prediction
#         nn_predictions = (nn_model.predict(X_scaled) > 0.5).astype("int32")
#         nn_fraud_count = np.sum(nn_predictions)  # Counting number of frauds predicted by NN

#         return {
#             "rf_fraud_count": int(rf_fraud_count),
#             "rf_predictions": rf_predictions.tolist(),
#             "nn_fraud_count": int(nn_fraud_count),
#             "nn_predictions": nn_predictions.flatten().tolist(),
#         }

#     except Exception as e:
#         return {"error": str(e)}

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import io
from model.model import load_trained_model, evaluate_model
from model.LogisticRegression import load_trained_model_Log
from model.test1 import load_trained_models
from log_test3 import load_logistic_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import re
import os;
import logging
import io
import joblib
import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
)
import tensorflow as tf

GENAI_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GENAI_API_KEY)

FRAUD_KEYWORDS = [
    "fraud", "scam", "illegal transaction", "money laundering",
    "unauthorized", "phishing", "card fraud", "chargeback fraud",
    "fraudulent transaction", "financial crime"
]


# Load models
lr1_model = joblib.load("logistic_model_test3.pkl")
rf_model = load_trained_model()  # Random Forest model
lr_model = load_trained_model_Log()  # Logistic Regression model
nn_model = load_trained_models()  # Neural Network model (Deep Learning)
# nn_model_1 = tf.keras.models.load_model("nn_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
scaler1 = joblib.load("scaler_20250316_000541.pkl")
nn_model_1 = joblib.load("nn_fraud_model_20250316_000541.pkl")

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] if you want to restrict origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API!"}


def is_fraud_related(query):
    """Checks if the query contains fraud-related keywords."""
    query_lower = query.lower()
    return any(re.search(r"\b" + keyword + r"\b", query_lower) for keyword in FRAUD_KEYWORDS)

@app.post("/detect_fraud/")
async def detect_fraud(query: str):
    """Receives query from frontend, processes it, and returns response."""
    query = query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not is_fraud_related(query):
        return {"response": "Unauthorized query"}

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        
        # Clean response: remove newlines, asterisks, and extra spaces
        paragraph_response = re.sub(r"[\*\n]", " ", response.text).strip()

        return {"response": paragraph_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
@app.post("/predict/rf/")
async def predict_rf(file: UploadFile = File(...)):
    """Predict fraud using the Random Forest model."""
    try:
        content = await file.read()
        file_ext = file.filename.split(".")[-1]

        # Read file
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format! Upload a CSV or Excel file."}

        df.dropna(inplace=True)

        # Ensure that 'Class' or any other target column is not in the input data
        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        # Preprocess and predict using Random Forest
        X_scaled = scaler.transform(df)
        rf_predictions = rf_model.predict(X_scaled)

        # Count fraud predictions
        rf_fraud_count = np.sum(rf_predictions)

        return {"rf_fraud_count": int(rf_fraud_count), "rf_predictions": rf_predictions.tolist()}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/lr/")
async def predict_lr(file: UploadFile = File(...)):
    """Predict fraud using the Logistic Regression model."""
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        df.dropna(inplace=True)
        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        X_scaled = scaler.transform(df)
        lr_predictions = lr_model.predict(X_scaled)

        lr_fraud_count = np.sum(lr_predictions)

        return {"lr_fraud_count": int(lr_fraud_count), "lr_predictions": lr_predictions.tolist()}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/nn/")
async def predict_nn(file: UploadFile = File(...)):
    """Predict fraud using the Neural Network model (Deep Learning)."""
    try:
        content = await file.read()
        file_ext = file.filename.split(".")[-1]

        # Read the uploaded file
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format! Upload a CSV or Excel file."}

        # Drop any rows with missing values
        df.dropna(inplace=True)

        # Preprocess the data: Drop the 'Class' column if it's present
        if 'Class' in df.columns:
            df.drop(columns=['Class'], inplace=True)

        # Ensure the column order matches the training data
        # You may need to reorder columns here if necessary (e.g., column_order)
        # Example: df = df[training_column_order]

        # Scale the features using the same scaler used during training
        X_scaled = scaler.transform(df)

        # Extract the model from the loaded tuple
        nn_model = load_trained_models()[0]  # Ignore the second part of the tuple

        # Neural Network Prediction
        nn_predictions = (nn_model.predict(X_scaled) > 0.5).astype("int32")
        nn_fraud_count = np.sum(nn_predictions)

        return {"nn_fraud_count": int(nn_fraud_count), "nn_predictions": nn_predictions.flatten().tolist()}

    except Exception as e:
        return {"error": str(e)}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/log/")
async def predict_rf(file: UploadFile = File(...)):
    """Predict fraud using the Logistic Regression model and return accuracy details."""
    try:
        content = await file.read()
        file_ext = file.filename.split(".")[-1]

        # Read file based on format
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format! Upload a CSV or Excel file."}

        df.dropna(inplace=True)

        # Ensure the 'Class' column is in the dataset
        if 'Class' not in df.columns:
            return {"error": "The input file must contain a 'Class' column."}

        # Separate features and labels
        X_test = df.drop(columns=['Class'])
        y_test = df['Class']

        # Preprocess data
        X_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = lr1_model.predict(X_scaled)

        # Calculate accuracy and fraud predictions
        test_accuracy = np.round((y_pred == y_test).mean(), 4)
        fraud_detected_test = int(np.sum(y_pred))
        total_fraud_present_test = int(np.sum(y_test))

        # Print logs for debugging
        print(f"🔹 Test Accuracy: {test_accuracy:.4f}")
        print(f"🔹 Fraud Predictions in Test Set: {fraud_detected_test}/{total_fraud_present_test}")

        return {
            "test_accuracy": test_accuracy,
            "fraud_detected_test": fraud_detected_test,
            "total_fraud_present_test": total_fraud_present_test
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_nn/")
async def evaluate_nn(file: UploadFile = File(...)):
    """Evaluate the neural network model and return classification reports and confusion matrices."""
    try:
        content = await file.read()
        file_ext = file.filename.split(".")[-1]

        # Read file
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format! Upload a CSV or Excel file."}

        df.dropna(inplace=True)

        # Extract labels (Assuming 'Class' column is target)
        y_test = df["Class"]
        X_test = df.drop(columns=["Class"])

        # Scale data
        X_scaled = scaler1.transform(X_test)

        # Make predictions
        y_pred_proba = nn_model_1.predict(X_scaled).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": cm
        }

    except Exception as e:
        logging.error(f"❌ Error evaluating model: {e}")
        return {"error": str(e)}

# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



@app.post("/evaluate/")
async def evaluate(file: UploadFile = File(...)):
    """Evaluate models and return classification reports and confusion matrices."""
    try:
        content = await file.read()
        file_ext = file.filename.split(".")[-1]

        # Read file
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format! Upload a CSV or Excel file."}

        df.dropna(inplace=True)

        # Extract labels (Assuming 'Class' column is target)
        y_test = df["Class"]
        X_test = df.drop(columns=["Class"])

        # Scale data
        X_scaled = scaler.transform(X_test)

        # Get evaluation results
        results = evaluate_model(X_scaled, y_test)
        return results

    except Exception as e:
        return {"error": str(e)}

logging.basicConfig(
    filename="server.log",  # Save logs to file
    level=logging.INFO,     # Set logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@app.post("/train_model/")
async def train_model(file: UploadFile = File(...)):
    """Receives a CSV file, trains a model, and returns accuracy results."""
    
    try:
        logging.info(f"📂 Received file: {file.filename}")

        # ✅ Save uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        logging.info(f"✅ File saved at: {file_location}")

        # ✅ Load dataset
        credit_card_data = pd.read_csv(file_location)

        # ✅ Remove temporary file after loading
        os.remove(file_location)

        logging.info("✅ File loaded successfully. Checking columns...")

        # ✅ Ensure the file contains 'Class' column
        if "Class" not in credit_card_data.columns:
            logging.error("❌ Missing 'Class' column in CSV file.")
            raise HTTPException(status_code=400, detail="CSV must contain a 'Class' column.")

        # ✅ Separate fraud (1) and legitimate (0) transactions
        legit = credit_card_data[credit_card_data.Class == 0]
        fraud = credit_card_data[credit_card_data.Class == 1]

        logging.info(f"🔹 Legit Transactions: {len(legit)}, Fraud Transactions: {len(fraud)}")

        # ✅ Ensure enough legitimate transactions for balancing
        if len(legit) < len(fraud):
            logging.error("❌ Not enough legitimate transactions to balance dataset.")
            raise HTTPException(status_code=400, detail="Not enough legitimate transactions to balance the dataset.")

        # ✅ Balance the dataset
        legit_sample = legit.sample(n=len(fraud), random_state=42)
        new_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

        # ✅ Split features and target
        X = new_df.drop(columns='Class', axis=1)
        Y = new_df['Class']

        # ✅ Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ✅ Save the scaler for later use
        joblib.dump(scaler, "scaler.pkl")

        # ✅ Split into training & testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

        logging.info("🚀 Training Logistic Regression model...")

        # ✅ Train Logistic Regression model
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, Y_train)

        # ✅ Save the trained model
        joblib.dump(model, "logistic_model.pkl")

        logging.info("✅ Model trained and saved successfully.")

        # ✅ Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        test_accuracy = accuracy_score(Y_test, test_predictions)

        # ✅ Count fraud predictions
        fraud_detected_test = sum(test_predictions)
        total_fraud_present_test = sum(Y_test)

        logging.info(f"📊 Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"🕵️ Fraud Detected in Test Set: {fraud_detected_test}/{total_fraud_present_test}")

        # ✅ Return results to frontend
        return {
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "fraud_detected_test": fraud_detected_test,
            "total_fraud_present_test": total_fraud_present_test
        }

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
