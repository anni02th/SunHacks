from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained models
models = []
for i in range(8):
    model_filename = f"linear_regression_model_sensor_{i + 1}.pkl"
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
        models.append(model)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    json_data = request.json
    
    # Convert JSON data to DataFrame
    data = pd.DataFrame(json_data)
    
    # Predict temperature for each sensor
    predictions = [model.predict(data) for model in models]
    
    # Convert predictions to list
    predictions_list = [prediction.tolist() for prediction in predictions]
    
    # Prepare response JSON
    response = {
        f"sensor_{i + 1}": prediction for i, prediction in enumerate(predictions_list)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
