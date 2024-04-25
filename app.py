from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("dataset.csv")

# Convert the timestamp to datetime object
df['ts'] = pd.to_datetime(df['ts'])

# Define features and target
X = df[['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8']]
y = df[['temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    timestamp = data['timestamp']
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    features = df[df['ts'] == timestamp][['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8']]
    prediction = model.predict(features)
    humidity = prediction[0][0]
    temperature = prediction[0][1]
    return jsonify({'humidity': humidity, 'temperature': temperature})

if __name__ == '_main_':
    app.run(debug=True)