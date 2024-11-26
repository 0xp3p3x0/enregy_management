import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


file_path = 'CSVma.csv'
data = pd.read_csv(file_path)
data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
data['min'] = pd.to_numeric(data['Min_Consumption'], errors='coerce')
data['avg'] = pd.to_numeric(data['Avg_Consumption'], errors='coerce')
data['max'] = pd.to_numeric(data['Max_Consumption'], errors='coerce')

X = data[['DateTime', 'min', 'max']]
y = data['avg']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data for TensorFlow model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a TensorFlow Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Predict on test set
y_pred_tf = model.predict(X_test_scaled).flatten()

# Evaluate the TensorFlow model
tf_mae = mean_absolute_error(y_test, y_pred_tf)
tf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tf))
tf_r2 = r2_score(y_test, y_pred_tf)

# Scikit-Learn Linear Regression as baseline
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate the Scikit-Learn model
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

# Summarize results
results = {
    "TensorFlow Neural Network": {
        "MAE": tf_mae,
        "RMSE": tf_rmse,
        "R^2": tf_r2
    },
    "Linear Regression (Scikit-Learn)": {
        "MAE": lr_mae,
        "RMSE": lr_rmse,
        "R^2": lr_r2
    }
}

import ace_tools as tools; tools.display_dataframe_to_user(name="Model Evaluation Results", dataframe=pd.DataFrame(results))
