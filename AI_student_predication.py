import pandas as pd
import numpy as np

print("                        AI Student Performance Prediction System")
print("                        ----------------------------------------")

# Load dataset
data = pd.read_csv("student_performance_data.csv")

print("\nDataset loaded successfully!")
print("\nFirst 5 records:")
print(data.head())

print("\nDataset information:")
print(data.info())

# ==============================
#  DATA ANALYSIS
# ==============================

print("\n===== DATA ANALYSIS =====")

# Show first 5 students
print("\nFirst 5 Records:")
print(data.head())

# Show number of rows and columns
print("\nDataset Shape:")
print(data.shape)

# Show column names
print("\nColumns:")
print(data.columns)

# Basic statistics
print("\nStatistical Summary:")
print(data.describe())

# Average values
print("\nAverage Values:")
print(data.mean(numeric_only=True))

# Highest final marks
print("\nHighest Final Marks:")
print(data["final_marks"].max())

# Lowest final marks
print("\nLowest Final Marks:")
print(data["final_marks"].min())

# Average final marks
print("\nAverage Final Marks:")
print(data["final_marks"].mean())




# ==============================
# DATA VISUALIZATION
# ==============================

import matplotlib.pyplot as plt

# Study Hours vs Final Marks
plt.scatter(data["study_hours"], data["final_marks"])

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")

plt.show()



# ==============================
#  AI MODEL TRAINING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Features
X = data[[
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_marks"
]]

# Target
y = data["final_marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create AI model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\n===== AI MODEL TRAINED SUCCESSFULLY =====")

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))






# ==============================
#  STUDENT PERFORMANCE PREDICTION
# ==============================

print("\n===== STUDENT PERFORMANCE PREDICTION =====")

study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
assignment_score = float(input("Enter assignment score: "))
previous_marks = float(input("Enter previous marks: "))

# Create input for AI model
new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "assignment_score": [assignment_score],
    "previous_marks": [previous_marks]
})

# Predict final marks
prediction = model.predict(new_student)

predicted_marks = prediction[0]

print("\nPredicted Final Marks:", round(predicted_marks, 2))

# Performance category
if predicted_marks >= 80:
    performance = "Excellent"
elif predicted_marks >= 60:
    performance = "Good"
elif predicted_marks >= 40:
    performance = "Average"
else:
    performance = "Needs Improvement"

print("Predicted Performance:", performance)