import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from datetime import datetime

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

df_load = pd.read_csv('LP.csv')


df_total = pd.read_excel('total.xlsx')

## pre_process total load data

df_total['DateTime'] = df_total.apply(lambda row: convert_to_24hr(row['Date'],row['Time'], row['TZ']), axis=1)
df_total = df_total.drop(columns=['Date','Time'])
df_total = df_total.dropna(subset=['DateTime']).reset_index(drop=True)
df_total.drop('TZ', axis=1, inplace=True)
print(df_total)

hour_data = df_total.resample('h', on='DateTime').sum()
print(hour_data.index)
hour_data = hour_data.reset_index()
hour_data['load'] = hour_data.apply(lambda row: insert_load_data(df_load, row['DateTime']), axis=1)
print(hour_data)


# df_total.to_csv('total.csv', index=False)

## this function used to convert time to 24 hr format
