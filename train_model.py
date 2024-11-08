import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the dataset
file_path = 'C:/Users/SANYA GOEL/OneDrive/Desktop/New folder (4)/last.xlsx'  
df = pd.read_excel(file_path)

# 2. Define features and target
X = df[['precipitation(mm/day)', 'drainage(mm/day)', 'Elevation(m)', 'Water Table(coastal region)', 'urbanization', 'runoff coefficient']]
y = df['waterlogging (mm/day)']

# 3. Define numeric and categorical features
numerical_features = ['precipitation(mm/day)', 'drainage(mm/day)', 'Elevation(m)', 'runoff coefficient']
categorical_features = ['Water Table(coastal region)', 'urbanization']

# 4. Create transformers for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# 5. Create a column transformer that applies the correct transformer to each column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Create the pipeline: first apply preprocessing, then fit a RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 7. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the model pipeline on the training data
pipeline.fit(X_train, y_train)

# 9. Save the trained pipeline as a .pkl file
joblib.dump(pipeline, 'model_pipeline.pkl')
print("Model pipeline saved as 'model_pipeline.pkl'")

# 10. (Optional) Evaluate the model on the test set
y_pred = pipeline.predict(X_test)

# Print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Model Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Check accuracy within a 10% tolerance
tolerance = 0.10
relative_errors = np.abs((y_test - y_pred) / y_test)
accuracy_within_tolerance = np.mean(relative_errors <= tolerance) * 100
print(f'Accuracy within {tolerance*100:.0f}% tolerance: {accuracy_within_tolerance:.2f}%')
