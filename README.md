# Gravity Prediction Model

## Overview
This project implements a machine learning model to predict gravity values based on geographical and geophysical parameters. The model uses latitude, longitude, height, free-air correction (freeair), and Bouguer anomaly (bouguer) to predict gravity measurements in Nigeria.

## Dataset
The dataset (`Copy of Gravity_Datas_SW_Nigeria_Prof_LMOjigi(1).csv`) contains gravity measurements from Southwest Nigeria with the following features:
- PID: Point identifier
- l: Latitude coordinate
- f: Longitude coordinate
- h(m): Height in meters
- g(mgal): Gravity measurement in milligals
- FreeAir: Free-air correction values
- Bouguer: Bouguer anomaly values

## Dependencies
The project requires the following Python libraries:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit (for deployment)
```

## Model Development
The project follows these main steps:

1. **Data Loading and Exploration**
   - Loading the gravity measurement dataset
   - Initial data exploration and visualization

2. **Correlation Analysis**
   - Visualizing feature correlations using a heatmap
   - Identifying relationships between geographical features and gravity

3. **Data Preparation**
   - Splitting data into training (75%) and testing (25%) sets
   - Separating features from the target variable (gravity)

4. **Model Training**
   - Implementation of Linear Regression model
   - Training the model on the prepared dataset

5. **Model Evaluation**
   - Calculating Mean Squared Error (RMSE: 1.8)
   - Calculating Mean Absolute Error (MAE: 1.349)
   - Visualizing actual vs. predicted values

6. **Model Persistence**
   - Saving the trained model using joblib
   - Testing the saved model to ensure consistency

7. **Prediction Function**
   - Creating a function for making predictions on new inputs
   - Type-annotated parameters for better code documentation

## Usage

### Training the Model
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
data = pd.read_csv("Copy of Gravity_Datas_SW_Nigeria_Prof_LMOjigi(1).csv")

# Split data
X_train, X_test = train_test_split(data, random_state=42, test_size=0.25)

# Prepare features and target
train_input = X_train.drop(columns=['gravity', 's/no'])
test_input = X_test.drop(columns=['gravity', 's/no'])
train_target = X_train.gravity
test_target = X_test.gravity

# Train model
linear_reg = LinearRegression()
linear_reg.fit(train_input, train_target)

# Make predictions
prediction = linear_reg.predict(test_input)

# Evaluate model
mae = mean_absolute_error(test_target, prediction)
rmse = np.sqrt(mean_squared_error(test_target, prediction))
print(f'Mean Square Error is: {round(rmse,3)} and Mean Absolute Error is: {round(mae,3)}')
```

### Making Predictions
```python
def pred_input(latitude: float, longitude: float, height, freeair, bouguer):
    input_dict = {
        'latitude': latitude,
        'longitude': longitude,
        'height': height,
        'freeair': freeair,
        'bouguer': bouguer
    }
    
    input_df = pd.DataFrame([input_dict])
    prediction_single = linear_reg.predict(input_df)[0]
    
    return prediction_single

# Example prediction
gravity_value = pred_input(latitude=4.3, longitude=10.3, height=200, freeair=-20, bouguer=-2)
print(gravity_value)  # Output: 977883.6334437147
```

## Model Performance
The linear regression model achieves good performance with:
- RMSE (Root Mean Square Error): 1.8
- MAE (Mean Absolute Error): 1.349

The scatter plot of actual vs. predicted values shows a strong linear relationship, indicating that the model effectively captures the relationship between the geographical features and gravity measurements.

## Future Work
- Implement additional regression models (Decision Tree, KNN, etc.)
- Perform hyperparameter tuning using GridSearchCV
- Explore feature engineering to improve model performance
- Deploy the model as a web service using Streamlit
