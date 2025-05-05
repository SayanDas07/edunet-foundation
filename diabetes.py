# diabetes_analysis.py

# Import required libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (ensure 'diabetes.csv' is in the same directory or specify the path)
data = pd.read_csv('diabetes.csv')
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())




# Compute the correlation matrix
correlation_matrix = data.corr()


# Define feature matrix (X) and target variable (y)
X = data.drop(columns='Outcome')  # Features
y = data['Outcome']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'\nLogistic Regression Accuracy: {accuracy_logistic:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))

import joblib
# Save model and scaler
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")