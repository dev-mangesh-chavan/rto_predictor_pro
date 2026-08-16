🚀 RTO Predictor Pro | Enterprise ML Pipeline

An End-to-End Machine Learning Microservice to predict and prevent Cash-On-Delivery (COD) Return-to-Origin (RTO) losses in E-commerce.

🚨 The Business Problem

In the Indian E-commerce sector (e.g., Myntra, Flipkart), 30% to 40% of Cash-on-Delivery (COD) orders result in Return to Origin (RTO). Customers often place orders impulsively and refuse to accept them at delivery. This causes massive financial leakage in reverse logistics, wasted packaging, and blocked inventory. Manual verification of every order is impossible at scale.

💡 The AI Solution

RTO Predictor Pro is an automated ML pipeline that intercepts live checkout data, analyzes historical return patterns, and calculates an RTO Risk Score in milliseconds.
Based on the risk probability, it categorizes orders into actionable business buckets with Explainable AI (Reasoning):

🟢 Safe: Auto-Approve the order.

🟡 Review: High Risk (Send SMS link for ₹50 advance payment).

🔴 Blocked: Serial Returner (Auto-cancel order).

📊 Model Performance & Business Impact

The core engine is powered by an advanced Random Forest Classifier (150 estimators, max depth of 12 for complex non-linear patterns) trained on a robust dataset of 10,000 historical records and validated on 2,000 unseen test orders.

Overall Accuracy: 82.30%

ROC-AUC Score: 81.63% (Excellent pattern detection)

Precision (RTO): 89.00%

Business Impact: With 89% precision, when the AI flags an order as 'High Risk', it is correct 89% of the time. This allows the business to confidently ask for advance payments without falsely blocking genuine customers.

🏗️ System Architecture (Microservices)

The project follows an industry-standard decoupled architecture:

AI Engine (src/): Data validation and scaling logic + Random Forest inference logic + AI Reason Generator.

Backend API (main.py): A high-performance FastAPI server acting as the prediction microservice.

Frontend Dashboard (app.py): A Streamlit UI for real-time manual inference and bulk warehouse batch scanning.

Folder Structure

RTO-Predictor-Pro/
│
├── data/                         # Datasets (Training & Testing)
│   ├── raw_rto_data.csv          # 10,000 historical records
│   └── test_batch_100_orders.csv # 100 live orders for UI Batch Scan testing
│
├── models/                       # Serialized AI Assets
│   ├── rto_rf_model.pkl          
│   └── feature_scaler.pkl        
│
├── notebooks/                    # Core Scripts / Notebooks for R&D
│   ├── 01_data_generation.py     # Synthetic enterprise data generator (10k rows)
│   └── 02_model_training.py      # Model training & evaluation metrics
│
├── src/                          # Core Engine Logic
│   ├── data_processor.py         # Data validation, scaling & fail-safes
│   └── ml_predictor.py           # Inference & Explainable AI Reason Generation
│
├── main.py                       # FastAPI Backend Server
├── app.py                        # Streamlit Frontend Dashboard
├── requirements.txt              # Dependencies
└── README.md                     # Documentation


⚙️ How to Run the Project Locally

Step 1: Clone & Setup Environment

git clone https://github.com/yourusername/RTO-Predictor-Pro.git
cd RTO-Predictor-Pro

# Create and activate virtual environment
python -m venv renv

# For Windows:
.\renv\Scripts\activate
# For Mac/Linux:
source renv/bin/activate

# Install dependencies
pip install -r requirements.txt


Step 2: Start the FastAPI Backend (Terminal 1)

The backend engine must be running to serve predictions and generate logic.

python main.py


The API will be live at http://127.0.0.1:8000. You can test endpoints via Swagger UI at http://127.0.0.1:8000/docs.

Step 3: Start the Streamlit UI (Terminal 2)

Open a new terminal, activate the environment (.\renv\Scripts\activate), and launch the premium dashboard:

streamlit run app.py


The dashboard will automatically open in your browser.

🔥 Features

Single Order Inference: Real-time form to simulate a live user checkout. Outputs Risk %, Action, and the AI's logical reason.

Batch Processing (Warehouse Scan): Upload a CSV (like test_batch_100_orders.csv) to process hundreds of orders instantly using optimized vectorized Pandas logic.

Explainable AI: The model doesn't just give a score; it generates a human-readable reason (e.g., "High-value COD risk" or "Serial Returner detected").

Enterprise Error Handling: Built-in safeguards against empty CSVs, missing columns, and incorrect data types.

Developed as a robust, scalable Machine Learning solution for modern E-commerce challenges.