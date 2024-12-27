# Credit Card Fraud Detection Project

## Overview
This project implements a Credit Card Fraud Detection system using the Decision Tree algorithm. It includes a simple web interface built with Flask and HTML for user interaction.


## Features
- Detects fraudulent transactions using a Decision Tree model.
- User-friendly interface for submitting transaction data.
- Provides real-time predictions for fraud detection.



## Technologies Used
- **Python** (for backend logic and model training)
- **Flask** (for web framework)
- **HTML** (for user interface)
- **Pandas, Scikit-learn** (for data processing and machine learning)



## Installation

### Prerequisites
Ensure you have Python (>= 3.8) installed along with the following libraries:
- Flask
- Scikit-learn
- Pandas

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sereneooi/card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd card-fraud-detection
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```



## Usage
1. **Launch the Web Application**:
   Access the application via your web browser at `http://127.0.0.1:5000`.
2. **Enter Transaction Data**:
   Fill in the transaction details in the provided form.
3. **Submit for Prediction**:
   Click the "Predict" button to classify the transaction as "Fraudulent" or "Non-Fraudulent."

## Model Details
The Decision Tree model was trained using a labeled dataset of credit card transactions. Features include transaction amount, merchant details, and timestamp, among others.

### Performance Metrics
- Accuracy: **99.92822684699443%**

