# AI Fraud Detection System  

Fraud detection is a critical challenge in financial security, and this project pushes the boundaries by integrating **Machine Learning, Deep Learning, and Real-Time Monitoring** into a scalable system. The system leverages **Logistic Regression (as a baseline) and a Deep Neural Network (DNN)** to detect fraudulent transactions with high accuracy.  

🔍 **Trained on 70,000+ transactions**, this system is designed to detect fraudulent activities instantly using **Apache Kafka** for real-time streaming and an **AI-powered chatbot (Gemini AI)** for user interaction. 🚀  

---  
## 🔑 Key Features & Technologies Used  

✅ **Dual-Model Approach**: Logistic Regression for interpretability & Deep Learning (DNN) for high precision.  
✅ **Real-time Transaction Monitoring**: Integrated **Apache Kafka** for instant fraud detection & analysis.  
✅ **AI Chatbot (Gemini AI)**: Provides users with fraud insights and support. 🤖💬  
✅ **SMOTE (Synthetic Minority Over-sampling Technique)**: Handles class imbalance for robust performance.  
✅ **Comprehensive Model Evaluation**: Accuracy, AUC-ROC, Precision, Recall, and F1-Score.  
✅ **Frontend Visualization**: **React.js Dashboard** displaying classification reports & confusion matrix.  

---  
## 💡 Tech Stack Used  

- **Python** | **TensorFlow** | **Keras** | **Scikit-learn** | **Imbalanced-learn**  
- **Apache Kafka** (Producer-Consumer Model) for real-time fraud detection  
- **Gemini AI** | **Flask/FastAPI** for AI-driven chatbot  
- **React.js | JavaScript** for frontend dashboard visualization  

---  
## 🚀 Installation Guide  

### Prerequisites  
Before running the project, ensure you have the following installed:  
- Python (3.8 or later)  
- Apache Kafka (for real-time fraud detection)  
- Node.js (for running the React frontend)  

### 1️⃣ **Clone the Repository**  
```sh  
git clone https://github.com/Prince200510/AI-Fraud-Detection.git  
cd AI-Fraud-Detection  
```

### 2️⃣ **Set Up Apache Kafka**  
Kafka is required for real-time fraud detection. Install and start Kafka with:  
```sh  
# Start Zookeeper (if not already running)
bin/zookeeper-server-start.sh config/zookeeper.properties  

# Start Kafka Server  
bin/kafka-server-start.sh config/server.properties  
```

### 3️⃣ **Install Python Dependencies**  
```sh  
pip install -r server/requirements.txt  
```

### 4️⃣ **Run the Backend (FastAPI/Flask Server)**  
```sh  
cd server  
python app.py  
```

### 5️⃣ **Run Kafka Producer & Consumer**  
```sh  
python producer.py  # Sends transactions to Kafka  
python consumer.py  # Listens for fraud detection results  
```

### 6️⃣ **Run the React Frontend**  
```sh  
cd client  
npm install  
npm start  
```

---  
## 📌 Folder Structure  
```
AI-Fraud-Detection/
│── client/          # React.js frontend
│── server/          # FastAPI backend with AI models
│── models/          # Trained neural network models
│── kafka/           # Kafka producer & consumer scripts
│── requirements.txt # Python dependencies
│── README.md        # Project documentation
```

---  
## 📢 Contributing  
Feel free to fork this repository, submit issues, and send pull requests! Contributions are always welcome. 😊  

📧 **Contact:** [LinkedIn](https://www.linkedin.com/in/prince-maurya-810b83277/)  

---  
## ⭐ Acknowledgments  
Special thanks to **OpenAI, TensorFlow, and Apache Kafka** for providing powerful tools to make this project possible. 🚀
