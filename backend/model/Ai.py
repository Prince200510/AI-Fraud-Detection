import google.generativeai as genai
import re

# Configure Google Gemini API
GENAI_API_KEY = "AIzaSyAOoMvnyYwVMleBQU_JdxQrjRQ4uRh6wf0"
genai.configure(api_key=GENAI_API_KEY)

# Define fraud-related keywords
FRAUD_KEYWORDS = [
    "fraud", "scam", "illegal transaction", "money laundering",
    "unauthorized", "phishing", "card fraud", "chargeback fraud",
    "fraudulent transaction", "financial crime"
]

# Function to check if input is related to fraud detection
def is_fraud_related(query):
    query_lower = query.lower()
    return any(re.search(r"\b" + keyword + r"\b", query_lower) for keyword in FRAUD_KEYWORDS)

# Function to detect fraud in user input
def detect_fraud(user_query):
    user_query = user_query.strip()

    # Validate input
    if not user_query:
        return {"error": "Query cannot be empty"}

    # Check if query is fraud-related
    if not is_fraud_related(user_query):
        return {"response": "Unauthorized query"}

    # Generate response using Gemini API
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(user_query)

    return {"response": response.text}

# Take user input and process fraud detection
if __name__ == "__main__":
    user_input = input("Enter your query: ")  # Take input from user
    result = detect_fraud(user_input)  # Call function to process input
    print("Output:", result)  # Display output in console


