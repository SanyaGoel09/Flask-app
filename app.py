from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load('model_pipeline.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Log received data for debugging
    print("Received data:", data)

    # Convert data into DataFrame
    input_data = pd.DataFrame([data])

    # Check if columns match what the model expects (optional)
    expected_columns = [
        'precipitation(mm/day)', 'drainage(mm/day)', 'Elevation(m)', 
        'Water Table(coastal region)', 'urbanization', 'runoff coefficient'
    ]
    
    # Ensure the input data matches the expected columns
    if not all(col in input_data.columns for col in expected_columns):
        return jsonify({
            'status': 'error',
            'message': 'Input data does not match expected format'
        }), 400  # Bad Request

    # Make prediction
    try:
        # Perform prediction using the trained model pipeline
        prediction = model.predict(input_data)[0]
        response = {
            'waterlogging_prediction': prediction,
            'status': 'success'
        }
    except Exception as e:
        response = {
            'status': 'error',
            'message': str(e)
        }
    
    # Return prediction as JSON response
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
