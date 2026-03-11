# ✈️ Flight Delay Prediction System

An **End-to-End Machine Learning Project** that predicts whether a flight will be delayed by more than **15 minutes** using historical aviation data.

This project demonstrates the complete ML workflow:

Data → Cleaning → Feature Engineering → Model Training → Evaluation → Deployment

---

# 🚀 Project Overview

Airline delays affect millions of passengers every year.  
This project builds a **Machine Learning system** that predicts whether a flight will be delayed.

The model predicts:

0 → Flight On Time  
1 → Flight Delayed (More than 15 minutes)

---

# 📊 Dataset

Source:  
U.S. Bureau of Transportation Statistics (BTS)

Dataset includes:

- Flight Date
- Airline Carrier
- Departure Delay
- Arrival Delay
- Distance
- Taxi Out Time
- Weather Delay
- Carrier Delay

After cleaning the dataset contains **500,000+ flight records**.

---

# 🧠 Machine Learning Approach

### Target Variable

ARR_DEL15

Meaning:

1 → Delayed more than 15 minutes  
0 → On time

---

### Features Used

- MONTH
- DAY_OF_WEEK
- DISTANCE
- TAXI_OUT
- OP_UNIQUE_CARRIER

Airlines were converted using **One-Hot Encoding**.

---

# 🤖 Model Used

Random Forest Classifier

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example Performance:

Accuracy ≈ 78%  
ROC-AUC ≈ 0.64

---

# 🏗 Project Structure

```
flight_delay_project
│
├── app.py
├── main.py
├── predict.py
├── flight_delay_model.pkl
├── model_features.pkl
├── requirements.txt
├── Dockerfile
└── T_ONTIME_REPORTING.csv
```

---

# 💻 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Docker
- Git
- GitHub

---

# ⚙️ Installation

### 1 Clone the repository

```
git clone https://github.com/yourusername/flight-delay-predictor.git
```

```
cd flight-delay-predictor
```

---

### 2 Create Virtual Environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3 Install Requirements

```
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Train the Model

```
python main.py
```

This will:

- Clean the dataset
- Train the model
- Save the trained model

Generated files:

```
flight_delay_model.pkl
model_features.pkl
```

---

## Run Prediction from Terminal

```
python predict.py
```

Example Output:

```
Flight will be ON TIME (Probability: 0.82)
```

---

# 🌐 Run Web Application

Start Streamlit App:

```
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

Enter flight details and get prediction.

---

# 🐳 Run Using Docker

Build Docker Image

```
docker build -t flight-delay-app .
```

Run Docker Container

```
docker run -p 8501:8501 flight-delay-app
```

Open:

```
http://localhost:8501
```

---

# 📈 Example Prediction

```
Flight likely ON TIME
Probability: 1.00
```

Top Important Features:

- DISTANCE
- TAXI_OUT
- DAY_OF_WEEK
- AIRLINE

---

# 🔮 Future Improvements

- Add XGBoost model
- Cloud Deployment (AWS / GCP)
- Add API using FastAPI
- Add MLflow experiment tracking
- Implement full MLOps pipeline

---

# 👨‍💻 Author

**Syed Sajid Hussain**

Machine Learning Enthusiast  
SMIT Student

---

⭐ If you like this project please give it a **star on GitHub**
