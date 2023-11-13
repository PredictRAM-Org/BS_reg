import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
import os

# Function to calculate real income statement item
def calculate_real_income_statement_item(item, input2):
    real_item = item / (1 + (input2 / 100))
    return real_item

# Check if a JSON data file exists and is valid
data_file = 'data.json'
if os.path.isfile(data_file):
    try:
        with open(data_file, 'r') as file:
            data = json.load(file)
        item_data = data.get('item_data')
        input2_data = data.get('input2_data')
    except (json.JSONDecodeError, KeyError):
        print("Data file error. Creating a sample data JSON file.")
        item_data = [100, 110, 120, 130, 140]  # Sample item data
        input2_data = [5, 6, 7, 8, 9]  # Sample input2 data
else:
    print("Data file not found. Creating a sample data JSON file.")
    item_data = [100, 110, 120, 130, 140]  # Sample item data
    input2_data = [5, 6, 7, 8, 9]  # Sample input2 data
    data = {
        'item_data': item_data,
        'input2_data': input2_data
    }
    with open(data_file, 'w') as file:
        json.dump(data, file)

# Create DataFrame for income statement item and input2 data
item_df = pd.DataFrame({'Income Statement Item': item_data})
input2_df = pd.DataFrame({'Input2': input2_data})

# Perform linear regression
model = LinearRegression()
model.fit(input2_df, item_df)

# Function to predict income statement item based on a percentage change in input2
def predict_income_statement_item(input2_percentage_change):
    # Calculate the new input2 value after the percentage change
    new_input2 = input2_data[-1] * (1 + input2_percentage_change / 100)

    # Predict income statement item based on the new input2 value
    predicted_item = model.predict([[new_input2]])[0][0]

    return predicted_item

try:
    # Input percentage change in input2 from the user
    input2_percentage_change = float(input("Enter the percentage change in Input2 (e.g., 5 for 5% increase, -3 for 3% decrease): "))
    predicted_item = predict_income_statement_item(input2_percentage_change)
    print(f"Predicted Income Statement Item for the Next Quarter with {input2_percentage_change}% Change in Input2: {predicted_item:.2f}")

    # Plot the predicted item point on the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(item_data)), item_data, label='Historical Data', color='blue')
    plt.scatter(len(item_data), predicted_item, label='Predicted Income Statement Item', color='red', marker='x', s=100)
    plt.plot(range(len(item_data)), model.predict(input2_df), label='Regression Line', color='green')
    plt.title('Input2 vs. Income Statement Item')
    plt.xlabel('Quarters')
    plt.ylabel('Income Statement Item')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph.png')  # Save the graph as 'graph.png'
    plt.show()
except ValueError:
    print("Invalid input. Please enter a valid numeric value for the percentage change.")
