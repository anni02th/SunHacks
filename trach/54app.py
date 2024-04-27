import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to load data from multiple CSV files in a folder
def load_data_from_month_folder(month_folder_path):
    # Initialize an empty list to store DataFrames
    dfs = []
    # Iterate over each file in the month folder
    for filename in os.listdir(month_folder_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            # Construct the full path to the file
            file_path = os.path.join(month_folder_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            dfs.append(df)
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_month_df = pd.concat(dfs, ignore_index=True)
    return combined_month_df

# Path to the folder containing the month folders
main_folder_path = "Training Dataset"

# Initialize an empty list to store DataFrames for each month
month_dfs = []

# Iterate over each folder in the main folder
for month_folder in os.listdir(main_folder_path):
    month_folder_path = os.path.join(main_folder_path, month_folder)
    # Check if the folder is a directory
    if os.path.isdir(month_folder_path):
        # Load data from the month folder and append it to the list
        month_df = load_data_from_month_folder(month_folder_path)
        month_dfs.append(month_df)

# Concatenate data from all months into a single DataFrame
combined_data = pd.concat(month_dfs, ignore_index=True)

# Data Preprocessing
# Assuming the target variables are 'temperature_sensor_1', 'temperature_sensor_2', ..., 'temperature_sensor_8'
y = combined_data[['temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]

# Features
X = combined_data.drop(columns=['temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8'])

# Exclude non-numeric columns from the feature set
non_numeric_columns = ['ts']  # Adjust this list based on your actual column names
X_numeric = X.drop(columns=non_numeric_columns)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model Training
models = []
for i in range(8):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train.iloc[:, i])
    models.append(model)

# Save the trained models to PKL files
for i, model in enumerate(models):
    model_filename = f"linear_regression_model_sensor_{i + 1}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# Model Evaluation
# Predicting on the testing set
y_preds = [model.predict(X_test_scaled) for model in models]

# Calculating evaluation metrics for each temperature sensor
maes = [mean_absolute_error(y_test.iloc[:, i], y_preds[i]) for i in range(8)]
mses = [mean_squared_error(y_test.iloc[:, i], y_preds[i]) for i in range(8)]
rmses = [mean_squared_error(y_test.iloc[:, i], y_preds[i], squared=False) for i in range(8)]

for i in range(8):
    print(f"Sensor {i + 1} - Mean Absolute Error:", maes[i])
    print(f"Sensor {i + 1} - Mean Squared Error:", mses[i])
    print(f"Sensor {i + 1} - Root Mean Squared Error:", rmses[i])