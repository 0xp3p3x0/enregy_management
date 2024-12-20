# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from datetime import datetime

# Function to convert time from 12-hour to 24-hour form
# Function to calculate load based on time intervals
def insert_load_data(load_data, ltime):
    
    load_len = load_data.shape[0]  # Get the number of rows in load_data
    load = 0.0  # Initialize load to zero
    localtime = datetime.strptime(ltime, '%Y-%m-%d %H:%M:%S').time()#ltime.time()  # Extract time from the current DateTime
    for i in range(load_len):
        starttime = datetime.strptime(load_data['start'][i], '%H:%M:%S').time()  # Convert start time to time object
        endtime  = datetime.strptime(load_data['end'][i], '%H:%M:%S').time()  # Convert end time to time object
        # Check if the current time falls within the start and end time
        if starttime < localtime and endtime > localtime:
            load += load_data['load'][i]  # Add the load value
    return load

# Load data from CSV and Excel files
df_load = pd.read_csv('LP.csv')  # Load load profile data
df_total = pd.read_csv('total_csv.csv')  # Load total data


#df_total = df_total.dropna(subset=['DateTime'])
#df_total = df_total.drop('Unnamed: 0')

print(df_total)  # Print the preprocessed total load data

#df_total.to_csv('total_csv.csv')

# Calculate the load for each hour based on time intervals in df_load
df_total['load'] = df_total.apply(lambda row: insert_load_data(df_load, row['DateTime']))
print(df_total)  # Print the hourly data with calculated load

# Save the processed data to a CSV file
#df_total.to_csv('total.csv', index=False)

# Detecting and visualizing outliers using a heatmap
plt.figure(figsize=(10, 5))  # Set the figure size for the plot
c = df_total.corr()  # Compute the correlation matrix
sns.heatmap(c, cmap="BrBG", annot=Fa)  # Plot the heatmap with annotations
plt.show()  # Display the heatmap
