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

# Check if 'AC_temperature' column is present in the dataset
if 'AC_temperature' not in combined_data.columns:
    print("Error: 'AC_temperature' column not found in the dataset.")
    # Handle the error gracefully, possibly by exiting the program or logging the issue
else:
    # Data Preprocessing
    # Assuming the target variable is 'AC_temperature'
    X = combined_data.drop(columns=['AC_temperature'])  # Features
    y = combined_data['AC_temperature']  # Target variable

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Save the trained model to a PKL file
    model_filename = "linear_regression_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    # Model Evaluation
    # Predicting on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculating evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
