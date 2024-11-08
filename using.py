import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load('model_pipeline.pkl')

# Example input data
example_query = {
    'precipitation(mm/day)': 20.0,
    'drainage(mm/day)': 5.0,
    'Elevation(m)': 4.0,
    'Water Table(coastal region)': 'Low',  
    'urbanization': 'High',  
    'runoff coefficient': 0.7
}

# Convert input to DataFrame
query_df = pd.DataFrame([example_query])

# Make prediction
predicted_value = pipeline.predict(query_df)[0]
print(f"Predicted waterlogging (mm/day): {predicted_value:.2f}")
