from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense
from keras._tf_keras.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#processing data
def convert_to_24hr(date_val,time_str, tz_str):
    
    hour = int(time_str.hour)
    if 'PM' in tz_str and hour != 12:
        hour += 12
    elif 'AM' in tz_str and hour == 12:
        hour = 0
    # Format back to HH:MM:SS  
    tmp_datetime = datetime(date_val.year, date_val.month, date_val.day, hour, time_str.minute, time_str.second)
    return tmp_datetime


def insert_load_data(load_data, ltime):
    load_len = load_data.shape[0]
    load = 0.0
    for i in range(load_len):
        localtime = ltime.time()
        starttime = datetime.strptime(load_data['start'][i], '%H:%M:%S').time()
        endtime  = datetime.strptime(load_data['end'][i], '%H:%M:%S').time()
        if starttime < localtime and endtime > localtime:
            load += load_data['load'][i]
    return load;


#load data
df_load = pd.read_csv('LP.csv')
df_total = pd.read_excel('total.xlsx')

df_total['DateTime'] = df_total.apply(lambda row: convert_to_24hr(row['Date'],row['Time'], row['TZ']), axis=1)
df_total = df_total.drop(columns=['Date','Time'])
df_total = df_total.dropna(subset=['DateTime']).reset_index(drop=True)
df_total.drop('TZ', axis=1, inplace=True)
print(df_total)

eda_data = df_total
eda_data['load'] = df_total.apply(lambda row: insert_load_data(df_load, row['DateTime']), axis=1)


##LSTM

train_data = df_total
train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])

train_data = train_data.sort_values('DateTime')
Y = train_data['avg'].values
wide = (Y.max() - Y.min()) 
minY = Y.min()
Y = (Y - minY) / wide # Normalize

X = train_data['DateTime'].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).tolist()
X = np.array(X, dtype=np.float32)  # Convert to a NumPy array of type float32

# Reshape X for LSTM input: (samples, timesteps, features)

X = X.reshape((X.shape[0], 1, X.shape[1]))

# Load or create model
if os.path.exists('model.h5'):
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss= MeanSquaredError() )
else:
    model = Sequential([
    LSTM(32, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    # Second LSTM layer
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(256, activation='relu', return_sequences=True),
    LSTM(512, activation='relu', return_sequences=True),
    LSTM(256, activation='relu', return_sequences=True),
    # Third (final) LSTM layer
    LSTM(64, activation='relu', return_sequences=False),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss= MeanSquaredError() )

#############################################################


#EDA analysis
def train_ai_model(data):
    data['hour'] = data['DateTime'].dt.hour
    X = data[['hour', 'min', 'max']]
    y = data['avg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred)
        }
   
    best_model_name = max(results, key=lambda k: results[k]['R²'])
    best_model = models[best_model_name]
    
    return best_model, results

def visualize(data, daily_data, stats):
    
    plt.subplot(1, 2, 1)
    plt.plot(daily_data.index, daily_data['avg'], label='Average Daily Consumption', marker='o')
    plt.axhline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axhline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.scatter(stats["Peak Demand Date"], stats["Peak Avg Consumption"], color='orange', label='Peak Demand', s=100)
    plt.xlabel('Date')
    plt.ylabel('Average Energy Consumption')
    plt.title('Daily Energy Consumption Trends')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    sns.histplot(daily_data['avg'], kde=True, bins=30, color='blue', edgecolor='black')
    plt.axvline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axvline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.xlabel('Average Daily Consumption')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Energy Consumption')
    plt.legend()
    
    plt.tight_layout()
    
    plt.show()
    
def descriptive_stats(data):
    daily_data = data.groupby(data['DateTime'].dt.date)['avg'].mean().reset_index()
    daily_data.columns = ['Date', 'avg']
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    stats = {
        "Mean Consumption": daily_data['avg'].mean(),
        "Median Consumption": daily_data['avg'].median(),
        "Peak Avg Consumption": daily_data['avg'].max(),
        "Peak Demand Date": daily_data.loc[daily_data['avg'].idxmax(), 'Date']
    }
    
    return stats, daily_data
#################################
root = Tk()


def corr():
    stats, daily_data = descriptive_stats(train_data)
    print("Descriptive Statistics:", stats)
    
    # Train AI-driven optimization model
    best_model, model_results = train_ai_model(train_data)
    print("Model Results:", model_results)
    print("Best Model:", best_model)
    
    # Visualize results
    visualize(train_data, daily_data, stats)
    return
def eda():
    plt.figure(figsize=(10, 5))  # Set the figure size for the plot
    c = df_total.corr()  # Compute the correlation matrix
    sns.heatmap(c, cmap="BrBG", annot=True)  # Plot the heatmap with annotations
    plt.show()
    return
def trainmodel():
    model.fit(X, Y, epochs=50, batch_size=32)
    messagebox.showinfo("Action", f"completed")
    model.save('model.h5')
    return
def predict():
    datetime_value = datetime_entry.get()
    try:
        # Validate datetime input
        new_datetime = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        input_features = np.array([[new_datetime.year, new_datetime.month, new_datetime.day,
                                new_datetime.hour, new_datetime.minute]], dtype=np.float32)

        # Reshape for LSTM: (samples, timesteps, features)
        input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

        # Predict
        predicted_value = model.predict(input_features)
        messagebox.showinfo("Action", f"predict power: {predicted_value[0][0] * wide + minY} Kw")
    except ValueError:
        messagebox.showerror("Error", "Invalid datetime format! Use YYYY-MM-DD HH:MM:SS")

    return

root.title("Energy Management System")
frm = ttk.Frame(root, padding=20)

frm.grid()
ttk.Button(frm,text="Correlation", command = corr).grid(column=1, row=0)
ttk.Button(frm,text="EDA analysis", command = eda).grid(column=2, row=0)
ttk.Button(frm,text="Train LSTM model", command = trainmodel).grid(column=3, row=0)
ttk.Button(frm,text="Predict", command = predict).grid(column=4, row=0)
ttk.Button(frm,text="Quit", command = root.destroy).grid(column=5, row=0)

ttk.Label(frm, text="Input Datetime : ").grid(column=1,row=1)
datetime_entry = ttk.Entry(frm, width=80)

datetime_entry.grid(column=1,row=2,columnspan=5,padx=5, pady=5)

root.mainloop()




