# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from datetime import datetime

# Function to convert time from 12-hour to 24-hour format
def convert_to_24hr(date_val, time_str, tz_str):
    # Extract the hour from the time string
    hour = int(time_str.hour)
    # Adjust the hour based on AM/PM
    if 'PM' in tz_str and hour != 12:
        hour += 12
    elif 'AM' in tz_str and hour == 12:
        hour = 0
    # Combine date and adjusted time into a datetime object
    tmp_datetime = datetime(date_val.year, date_val.month, date_val.day, hour, time_str.minute, time_str.second)
    return tmp_datetime

# Function to calculate load based on time intervals
def insert_load_data(load_data, ltime):
    load_len = load_data.shape[0]  # Get the number of rows in load_data
    load = 0.0  # Initialize load to zero
    for i in range(load_len):
        localtime = ltime.time()  # Extract time from the current DateTime
        starttime = datetime.strptime(load_data['start'][i], '%H:%M:%S').time()  # Convert start time to time object
        endtime  = datetime.strptime(load_data['end'][i], '%H:%M:%S').time()  # Convert end time to time object
        # Check if the current time falls within the start and end time
        if starttime < localtime and endtime > localtime:
            load += load_data['load'][i]  # Add the load value
    return load

# Load data from CSV and Excel files
df_load = pd.read_csv('LP.csv')  # Load load profile data
df_total = pd.read_excel('total.xlsx')  # Load total data

# Preprocess the total load data
# Convert 'Date' and 'Time' columns to a single 'DateTime' column in 24-hour format
df_total['DateTime'] = df_total.apply(lambda row: convert_to_24hr(row['Date'], row['Time'], row['TZ']), axis=1)
# Drop the original 'Date' and 'Time' columns as they are no longer needed
df_total = df_total.drop(columns=['Date', 'Time'])
# Remove rows with missing 'DateTime' values and reset the index
df_total = df_total.dropna(subset=['DateTime']).reset_index(drop=True)
# Drop the 'TZ' column as it has already been used for conversion
df_total.drop('TZ', axis=1, inplace=True)
print(df_total)  # Print the preprocessed total load data

# Resample the data to hourly intervals, summing up values for each hour
hour_data = df_total.resample('h', on='DateTime').sum()
print(hour_data.index)  # Print the index to verify resampling
hour_data = hour_data.reset_index()  # Reset the index after resampling

# Calculate the load for each hour based on time intervals in df_load
hour_data['load'] = hour_data.apply(lambda row: insert_load_data(df_load, row['DateTime']), axis=1)
print(hour_data)  # Print the hourly data with calculated load

# Save the processed data to a CSV file
hour_data.to_csv('total.csv', index=False)

# Detecting and visualizing outliers using a heatmap
plt.figure(figsize=(10, 5))  # Set the figure size for the plot
c = hour_data.corr()  # Compute the correlation matrix
sns.heatmap(c, cmap="BrBG", annot=True)  # Plot the heatmap with annotations
plt.show()  # Display the heatmap
