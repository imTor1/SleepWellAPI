from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
try:
    model = joblib.load("healthsleep.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/api/sleephealth', methods=['POST'])
def sleep_health():
    try:
        # Receive JSON data
        data = request.json
        print(f"Received JSON data: {data}")

        # Required input fields
        required_fields = ['age', 'gender', 'dietary_habits', 'sleep_disorders',
                           'medication_usage', 'sleep_quality', 'daily_steps', 'physical_activity_level']

        # Check for missing input data
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing input data'}), 400

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])

        # Map input data to expected formats
        # Convert numerical fields
        numerical_fields = ['age', 'sleep_quality', 'daily_steps']
        for field in numerical_fields:
            try:
                input_data[field] = float(input_data[field])
            except ValueError:
                return jsonify({'error': f'Invalid input for {field}. Must be a number.'}), 400

        # Map categorical fields to match training data
        # Gender
        gender_map = {'male': 0, 'female': 1}
        input_data['gender'] = input_data['gender'].map(gender_map)
        if input_data['gender'].isnull().any():
            return jsonify({'error': 'Invalid gender input'}), 400

        # Dietary Habits
        dietary_habits_map = {'unhealthy': 0, 'medium': 1, 'healthy': 2}
        input_data['dietary_habits'] = input_data['dietary_habits'].map(dietary_habits_map)
        if input_data['dietary_habits'].isnull().any():
            return jsonify({'error': 'Invalid dietary habits input'}), 400

        # Sleep Disorders
        sleep_disorders_map = {'no': 0, 'yes': 1}
        input_data['sleep_disorders'] = input_data['sleep_disorders'].map(sleep_disorders_map)
        if input_data['sleep_disorders'].isnull().any():
            return jsonify({'error': 'Invalid sleep disorders input'}), 400

        # Medication Usage
        medication_usage_map = {'no': 0, 'yes': 1}
        input_data['medication_usage'] = input_data['medication_usage'].map(medication_usage_map)
        if input_data['medication_usage'].isnull().any():
            return jsonify({'error': 'Invalid medication usage input'}), 400

        # Physical Activity Level
        physical_activity_map = {'low': 0, 'medium': 1, 'high': 2}
        input_data['physical_activity_level'] = input_data['physical_activity_level'].map(physical_activity_map)
        if input_data['physical_activity_level'].isnull().any():
            return jsonify({'error': 'Invalid physical activity level input'}), 400

        # One-Hot Encoding of Categorical Variables
        categorical_cols = ['gender', 'physical_activity_level', 'dietary_habits',
                            'sleep_disorders', 'medication_usage']
        input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

        # Prepare the input data in the correct order for the model
        # Note: Ensure that the features are in the same order as when the model was trained
        # List of features as per the trained model
        feature_order = ['age', 'sleep_quality', 'daily_steps',
                         'gender_1',  # Assuming 'gender' was one-hot encoded
                         'physical_activity_level_1', 'physical_activity_level_2',
                         'dietary_habits_1', 'dietary_habits_2',
                         'sleep_disorders_1', 'medication_usage_1']

        # Add missing columns with default value 0
        for col in feature_order:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match training data
        input_data = input_data[feature_order]

        # Convert DataFrame to numpy array
        input_array = input_data.values

        # Predict using the model
        try:
            prediction = model.predict(input_array)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return jsonify({'error': 'Model prediction failed'}), 500

        # Convert prediction to a Python int and return the result
        status = "ดี" if int(prediction[0]) == 1 else "ไม่ดี"
        return jsonify({'sleep_health_status': status}), 200

    except Exception as e:
        # Log error and return 500
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
