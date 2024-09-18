from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
try:
    model = joblib.load("health_sleep.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/api/sleephealth', methods=['POST'])
def sleep_health():
    try:
        # Log the incoming request data
        print(f"Received request: {request.data}")
        
        # Receive JSON data
        data = request.json
        print(f"Received JSON data: {data}")

        # Retrieve and validate inputs
        age = data.get('age')
        bedtime = data.get('bedtime')
        wake_up_time = data.get('wake_up_time')
        daily_steps = data.get('daily_steps')

        # Log individual fields to check their values
        print(f"age: {age}, bedtime: {bedtime}, wake_up_time: {wake_up_time}, daily_steps: {daily_steps}")

        if None in [age, bedtime, wake_up_time, daily_steps]:
            return jsonify({'error': 'Missing input data'}), 400

        try:
            age = int(age)
            bedtime = int(bedtime)
            wake_up_time = int(wake_up_time)
            daily_steps = int(daily_steps)
        except ValueError:
            return jsonify({'error': 'Invalid input data. Must be integers.'}), 400

        # Encoding gender
        gender_input = data.get('gender')
        if gender_input == 'male':
            gender = 0
        elif gender_input == 'female':
            gender = 1
        else:
            return jsonify({'error': 'Invalid gender input'}), 400

        # Encoding dietary habits
        dietary_habits_input = data.get('dietary_habits')
        if dietary_habits_input == 'unhealthy':
            dietary_habits = 0
        elif dietary_habits_input == 'medium':
            dietary_habits = 1
        elif dietary_habits_input == 'healthy':
            dietary_habits = 2
        else:
            return jsonify({'error': 'Invalid dietary habits input'}), 400

        # Encoding other fields
        sleep_disorders_input = data.get('sleep_disorders')
        sleep_disorders = 1 if sleep_disorders_input == 'yes' else 0

        medication_usage_input = data.get('medication_usage')
        medication_usage = 1 if medication_usage_input == 'yes' else 0

        # Encoding sleep quality
        sleep_quality_input = data.get('sleep_quality')
        if sleep_quality_input == 'low':
            sleep_quality = 0
        elif sleep_quality_input == 'medium':
            sleep_quality = 1
        elif sleep_quality_input == 'high':
            sleep_quality = 2
        else:
            return jsonify({'error': 'Invalid sleep quality input'}), 400

        # Encoding physical activity level
        physical_activity_input = data.get('physical_activity_level')
        if physical_activity_input == 'low':
            physical_activity_level = 0
        elif physical_activity_input == 'medium':
            physical_activity_level = 1
        elif physical_activity_input == 'high':
            physical_activity_level = 2
        else:
            return jsonify({'error': 'Invalid physical activity level input'}), 400

        # Prepare data for the model
        x = np.array([[age, gender, dietary_habits, sleep_disorders, medication_usage,
                       bedtime, wake_up_time, sleep_quality, daily_steps, physical_activity_level]])

        # Predict using the model
        try:
            prediction = model.predict(x)
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
