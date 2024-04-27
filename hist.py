import logging
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup

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
df = load_datasets([feb_folder, mar_folder])

# Define features and target
X = df[['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8']]
y = df[['temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Function to fetch weather data from the specified URL
def get_weather_data():
    url = "http://localhost:5173/weat.html"  # Replace with the URL of the HTML page containing weather data
    response = requests.get(url)
    
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract temperature and humidity
        temperature_element = soup.find("h1", class_="temp")
        humidity_element = soup.find("div", class_="humidity")

        if temperature_element and humidity_element:
            temperature = temperature_element.text.strip()
            humidity_text = humidity_element.text.strip()
            humidity = humidity_text.split(":")[1].strip()
            
            return temperature, humidity
        else:
            print("Temperature or humidity element not found.")
    else:
        print("Failed to fetch HTML content:", response.status_code)

# Update the 'predict' function to display the database directly
@app.route('/templates/Hist.html')
def show_hist():
    return render_template('Hist.html')
def display_database():
    # Filter the dataset for the desired timestamp
    timestamp = '2024-04-27T07:48:00'  # Replace with the desired timestamp
    features = df[df['ts'] == datetime.fromisoformat(timestamp)][['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8', 'temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]
    
    if features.empty:
        return jsonify({'error': 'No data available for the provided timestamp'}), 404
    
    # Render the Hist.html template with the database data
    return render_template('Hist.html', data=features.to_html())

if __name__ == '__main__':
    app.run(debug=True)