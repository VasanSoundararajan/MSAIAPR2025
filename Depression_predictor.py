import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

od.download("https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data", force=True)
data = pd.read_csv("./student-depression-dataset/student_depression_dataset.csv")

data.drop(['id', 'Gender', 'Age', 'City', 'Profession', 'Work Pressure', 'CGPA', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Work/Study Hours', 'Financial Stress'], axis=1, inplace=True)

data = pd.get_dummies(data, columns=['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'])

# Define features (X) and target variable (y)
X = data[['Academic Pressure', 'Study Satisfaction', 'Have you ever had suicidal thoughts ?_Yes', 'Family History of Mental Illness_Yes']]  # Select the relevant columns explicitly
y = data['Depression']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier (you can choose other models as well)
model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Function to get user input and make predictions
def predict_depression():
    academic_pressure = float(input("Enter Academic Pressure (0-5): "))
    study_satisfaction = float(input("Enter Study Satisfaction (0-5): "))
    suicidal_thoughts = int(input("Have you ever had suicidal thoughts? (0 for No, 1 for Yes): "))
    family_history = int(input("Family History of Mental Illness? (0 for No, 1 for Yes): "))

    user_input = pd.DataFrame({
        'Academic Pressure': [academic_pressure],
        'Study Satisfaction': [study_satisfaction],
        'Have you ever had suicidal thoughts ?_Yes': [suicidal_thoughts],
        'Family History of Mental Illness_Yes': [family_history]
    })

    prediction = model.predict(user_input)
    print(f"Depression Prediction: {prediction[0]}")  # 0 or 1

# Example usage:
predict_depression()
