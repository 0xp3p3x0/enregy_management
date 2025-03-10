from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime, timedelta
from decimal import Decimal
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
    return load

#untilites

def generate_dates(start_date, end_date=None):
    """
    Generate a list of dates between start_date and end_date.
    If end_date is None, return only the start_date.
    """
    if end_date:
        current_date = start_date
        date_list = []
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(hours=1)
        return date_list
    else:
        return [start_date]

#load data
df_load = pd.read_csv('LP.csv')
df_total = pd.read_excel('train.xlsx')
df_train = pd.read_excel('train.xlsx')

df_total['DateTime'] = df_total.apply(lambda row: convert_to_24hr(row['Date'],row['Time'], row['TZ']), axis=1)
df_total = df_total.drop(columns=['Date','Time'])
df_total = df_total.dropna(subset=['DateTime']).reset_index(drop=True)
df_total.drop('TZ', axis=1, inplace=True)
print(df_total)

df_train['DateTime'] = df_train.apply(lambda row: convert_to_24hr(row['Date'],row['Time'], row['TZ']), axis=1)
df_train = df_train.drop(columns=['Date','Time'])
df_train = df_train.dropna(subset=['DateTime']).reset_index(drop=True)
df_train.drop('TZ', axis=1, inplace=True)
print(df_train)


eda_data = df_total
eda_data['load'] = df_total.apply(lambda row: insert_load_data(df_load, row['DateTime']), axis=1)


##LSTM

train_data = df_train
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

def visualize(data, stats):
    
    plt.subplot(1, 2, 1)
    plt.plot(data['DateTime'], data['avg'], label='Average Daily Consumption', marker='o')
    plt.axhline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axhline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.scatter(stats["Peak Demand Date"], stats["Peak Avg Consumption"], color='orange', label='Peak Demand', s=100)
    plt.xlabel('Date')
    plt.ylabel('Average Energy Consumption')
    plt.title('Daily Energy Consumption Trends')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    sns.histplot(data['avg'], kde=True, bins=30, color='blue', edgecolor='black')
    plt.axvline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axvline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.xlabel('Average Daily Consumption')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Energy Consumption')
    plt.legend()
    
    plt.tight_layout()
    
    plt.show()
    
def descriptive_stats(data):
 
    stats = {
        "Mean Consumption": data['avg'].mean(),
        "Median Consumption": data['avg'].median(),
        "Peak Avg Consumption": data['avg'].max(),
        "Peak Demand Date": data.loc[data['avg'].idxmax(), 'DateTime']
    }
    
    tmp_data = data[['DateTime','avg','min','max']]
    
    return stats, tmp_data
#################################
root = Tk()


def corr():
    stats, data = descriptive_stats(df_total)
    print("Descriptive Statistics:", stats)
    
    # Train AI-driven optimization model
    best_model, model_results = train_ai_model(data)
    print("Model Results:", model_results)
    print("Best Model:", best_model)
    
    # Visualize results
    visualize(data, stats)
    return
def eda():
    plt.figure(figsize=(10, 5))  # Set the figure size for the plot
    c = df_total.corr()  # Compute the correlation matrix
    sns.heatmap(c, cmap="BrBG", annot=True)  # Plot the heatmap with annotations
    plt.show()
    return
def trainmodel():
    model.fit(X, Y, epochs=50, batch_size=256)
    messagebox.showinfo("Action", f"completed")
    model.save('model.h5')
    return
def predict():
    
    start_value = start_entry.get()  # Retrieve start date from entry
    end_value = end_entry.get()      # Retrieve end date from entry (optional)
    emp_flag = True

    try:
        # Validate and parse input datetime strings
        start_date = datetime.strptime(start_value, '%Y-%m-%d %H:%M:%S')
        end_date = None
        if end_value:
            end_date = datetime.strptime(end_value, '%Y-%m-%d %H:%M:%S')
        else:
            emp_flag = False

        # Generate the list of datetime values
        date_list = generate_dates(start_date, end_date)

        # Prepare features and predict for each datetime
        predictions = []
        times = []
        for current_date in date_list:
            input_features = np.array([[current_date.year, current_date.month, current_date.day,
                                         current_date.hour, current_date.minute]], dtype=np.float32)
            input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))
            
            predicted_value = model.predict(input_features)
            #predictions.append(predicted_value[0][0] * wide + minY)  # Scaled prediction
            times.append(current_date)
            
            formatted_value = "%.2f" % (abs(predicted_value[0][0] * wide + minY))
            predictions.append(Decimal(formatted_value))

        data = pd.DataFrame({'Datetime': times, 'Predictions': predictions})
        data['load'] = data.apply(lambda row: insert_load_data(df_load, row['Datetime']), axis=1)
        
        df = pd.DataFrame(data)
        output_path = 'predictions.xlsx'
        df.to_excel(output_path, index=False)
        # Display all predictions
        # result_message = "\n".join(predictions)
        
        if emp_flag:
            messagebox.showinfo("Predictions", f"done")
            
            plt.plot( df['Datetime'],  df['Predictions'], label='Average Daily Consumption', marker='o')
            plt.xlabel('Date')
            plt.ylabel('Average Energy Consumption')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
             messagebox.showinfo("Predictions", f"{data['Predictions'][0]}Kw")
        
    
    except ValueError:
        messagebox.showerror("Error", "Invalid datetime format! Use YYYY-MM-DD HH:MM:SS")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    # start_value = start_entry.get()
    # end_value = end_entry.get()
    # emp_flag = True
    # if(end_value == ""): emp_flag =False 
    # try:
    #     # Validate datetime input
    #     start_date = datetime.strptime(start_value, '%Y-%m-%d %H:%M:%S')
    #     end_date = None
    #     if end_value:
    #         end_date = datetime.strptime(end_value, '%Y-%m-%d %H:%M:%S')
    #         input_features_end = np.array([[end_date.year, end_date.month, end_date.day,
    #                                 end_date.hour, end_date.minute]], dtype=np.float32)
    #         input_features_end = input_features_end.reshape((input_features_end.shape[0], 1, input_features_end.shape[1]))
            
    #     input_features_start = np.array([[start_date.year, start_date.month, start_date.day,
    #                             start_date.hour, start_date.minute]], dtype=np.float32)

        

    #     # Reshape for LSTM: (samples, timesteps, features)
    #     input_features_start = input_features_start.reshape((input_features_start.shape[0], 1, input_features_start.shape[1]))

    #     # Predict
    #     if(emp_flag):
    #         predicted_value = model.predict(input_features_start)
    #         messagebox.showinfo("Action", f"predict power: {predicted_value[0][0] * wide + minY} Kw")
    #     else:
    #         predicted_value = model.predict(input_features_start)
    #         messagebox.showinfo("Action", f"predict power: {predicted_value[0][0] * wide + minY} Kw")
        
    # except ValueError:
    #     messagebox.showerror("Error", "Invalid datetime format! Use YYYY-MM-DD HH:MM:SS")

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
start_entry = ttk.Entry(frm)
end_entry = ttk.Entry(frm)

start_entry.grid(column=1,row=2,columnspan=3,padx=5, pady=5)
end_entry.grid(column=3,row=2,columnspan=3,padx=5, pady=5)

root.mainloop()




