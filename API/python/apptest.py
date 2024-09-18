from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("house_price_model.pkl")

@app.route('/api/house', methods=['POST'])
def house():

    
    age = int(request.form.get('age')) 
    distance = int(request.form.get('distance')) 
    minimart = int(request.form.get('minimart')) 
    
    # Prepare the input for the model
    x = np.array([[age, distance, minimart]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    return {'price': round(prediction[0], 2)}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)