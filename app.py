import logging
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from datetime import datetime
import os

app = Flask(__name__)

# Function to load datasets from multiple folders
def load_datasets(folder_paths):
    df_list = []
    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df['ts'] = pd.to_datetime(df['ts'])
                df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Load datasets from Feb and Mar folders
datasets_folder = "C:/Users/anike/OneDrive/Desktop/new run/SunHacks/Ordered Dataset"
feb_folder = os.path.join(datasets_folder, "February 2024")
mar_folder = os.path.join(datasets_folder, "March 2024")
api_folder = os.path.join(datasets_folder, "April 2024")
df = load_datasets([feb_folder, mar_folder,api_folder])

# Define features and target
X = df[['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8']]
y = df[['temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Function to calculate accuracy using cross-validation
def calculate_accuracy(X, y, model):
    # Perform cross-validation on the entire dataset
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')  # 5-fold cross-validation
    mean_mae = -scores.mean()
    return mean_mae

# Calculate accuracy
accuracy = calculate_accuracy(X, y, model)
print("Mean Absolute Error (MAE) using Cross-Validation:", accuracy)

@app.route('/')
def index():
    return render_template('pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    timestamp_str = request.form['timestamp']
    
    # Parse ISO 8601 timestamp
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return jsonify({'error': 'Invalid timestamp format'}), 400
    
    # Filter the dataset for the given timestamp
    features = df[df['ts'] == timestamp][['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8']]
    
    if features.empty:
        return jsonify({'error': 'No data available for the provided timestamp'}), 404
    
    # Predict temperature and humidity
    prediction = model.predict(features)
    humidity = prediction[0][0]
    temperature = prediction[0][1]
    
    # Print prediction to terminal
    print(f"Prediction for timestamp {timestamp}: Humidity - {humidity}, Temperature - {temperature}")
    
    return render_template('pred.html', humidity=humidity, temperature=temperature)

if __name__ == '__main__':
    app.run(debug=True)
