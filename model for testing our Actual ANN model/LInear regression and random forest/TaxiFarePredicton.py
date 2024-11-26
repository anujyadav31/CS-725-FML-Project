import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# import sklearn
# print(sklearn.__version__)



# Initialize the model
model = LinearRegression()



data_path = 'C:\\Users\\prabhat patel\\OneDrive\\Desktop\\fml project\\linear regression and random forest\\LInear regression and random forest\\data.csv'
data= pd.read_csv(data_path)
# df = pd.read_csv('data_path', low_memory=False)

#

data= data.dropna()
data= data[(data['fare_amount']>0) & (data['passenger_count']>0)]
# print(data.head())
# print(data.info())
# print(data.describe())

data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])

data['hour'] = data['tpep_pickup_datetime'].dt.hour
data['day'] = data['tpep_pickup_datetime'].dt.day
data['weekday'] = data['tpep_pickup_datetime'].dt.weekday
data['month'] = data['tpep_pickup_datetime'].dt.month

# print(data[['hour', 'day', 'weekday', 'month']].head())

data['is_shared_ride'] = data['passenger_count'] > 1
# print(data[['passenger_count', 'is_shared_ride']].head())

# data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60  # in minutes
data = data[data['trip_distance'] <= 100]

# plt.scatter(data['trip_distance'], data['fare_amount'], alpha=0.5)
# plt.title("Distance vs Fare")
# plt.xlabel("Distance (km)")
# plt.ylabel("Fare (USD)")
# plt.show()

# Define features and target
X = data[['trip_distance', 'hour', 'day', 'weekday', 'month']]  # Add other features if needed
y = data['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
# Train the model on the training data
model.fit(X_train, y_train)

# Check coefficients and intercept
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)


# plot

plt.scatter(y_test, y_pred, alpha=0.5)
plt.title("Actual vs Predicted Fares")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.show()

average_fare = y_test.mean()
print("Average Fare Amount:", average_fare)
percentage_error = (rmse / average_fare) * 100
print(f"RMSE as a Percentage of Average Fare: {percentage_error:.2f}%")



residuals = y_test - y_pred
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Actual Fares")
plt.xlabel("Actual Fare")
plt.ylabel("Residuals")
plt.show()


#random forest model 


# Initialize the model


# Import required libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Initialize the Random Forest Regressor
print("1")
rf_model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1,verbose=1)

# Train the model
print("2")
rf_model.fit(X_train, y_train)

# Make predictions on the test set
print("3")
y_pred_rf = rf_model.predict(X_test)

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Print the evaluation results
print("Random Forest - Mean Absolute Error (MAE):", mae_rf)
print("Random Forest - Root Mean Squared Error (RMSE):", rmse_rf)

feature_importance = rf_model.feature_importances_
for i, feature in enumerate(X_train.columns):
    print(f"Feature: {feature}, Importance: {feature_importance[i]:.4f}")



# Plot Actual vs Predicted for Linear Regression
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Linear Regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()

# Plot Actual vs Predicted for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6, label='Random Forest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()

# model tuning

# rf_model = RandomForestRegressor(random_state=42)

# defining hyper parameter grid

# param_grid = {
#     'n_estimators': [50, 100, 20],  # Number of trees
#     # 'max_depth': [10, 20, 30, None],  # Max depth of trees
#     # 'min_samples_split': [2, 5, 10],  # Min samples to split a node
#     # 'min_samples_leaf': [1, 2, 4],    # Min samples in a leaf node
#     # 'max_features': ['sqrt', 'log2'] # Features to consider at each split
# }
# print("4")
# # GridSearch method for hyper parameter tuning
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
#                            cv=3, scoring='neg_mean_squared_error', 
#                            verbose=10, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # getting the best parameter

# print("Best Hyperparameters:", grid_search.best_params_)

# # evaluation of tuned model

# best_model = grid_search.best_estimator_
# y_pred_tuned = best_model.predict(X_test)
# rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
# print("Tuned Model RMSE:", rmse_tuned)




