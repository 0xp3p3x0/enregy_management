import fileinput
import os
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense
from keras._tf_keras.keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
from datetime import datetime


train_data = pd.read_csv('total.csv')

if 'DateTime' not in train_data.columns or 'avg' not in train_data.columns:
    raise ValueError("The dataset must contain 'DateTime' and 'avg' columns.")

train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])

train_data = train_data.sort_values('DateTime')
Y = train_data['avg'].values
wide = (Y.max() - Y.min()) 
Y = (Y - Y.min()) / wide # Normalize

X = train_data['DateTime'].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).tolist()
X = np.array(X, dtype=np.float32)  # Convert to a NumPy array of type float32

# Reshape X for LSTM input: (samples, timesteps, features)

X = X.reshape((X.shape[0], 1, X.shape[1]))

# Load or create model
if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss= 'MSE' )



model.fit(X, Y, epochs=50, batch_size=32)

model.save('model.h5')


new_datetime = datetime(2024, 12, 11, 15, 30)
input_features = np.array([[new_datetime.year, new_datetime.month, new_datetime.day,
                            new_datetime.hour, new_datetime.minute]], dtype=np.float32)

# Reshape for LSTM: (samples, timesteps, features)
input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

# Predict
predicted_value = model.predict(input_features)


print(f"Predicted Value: {predicted_value * wide}")
