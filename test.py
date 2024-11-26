import pandas as pd

def convert_to_24hr(time_str, tz_str):
    time_parts = time_str.split(':')
    hour = int(time_parts[0])
    if 'AM' in tz_str and hour != 12:
        hour += 12
    elif 'PM' in tz_str and hour == 12:
        hour = 0
    # Format back to HH:MM:SS
    return f"{hour:02}:{time_parts[1]}:{time_parts[2]}"

# Apply the conversion function to the 'Time' column


excel_file = 'EQ.xlsx'
data = pd.read_excel(excel_file)

csv_EQ = 'CSV_EQ.csv'
data.to_csv(csv_EQ)

excel_file = 'hostel.xlsx'
data = pd.read_excel(excel_file)

csv_EQ = 'CSV_hostel.csv'
data.to_csv(csv_EQ)

excel_file = 'MA.xlsx'
data = pd.read_excel(excel_file)

csv_EQ = 'CSV_MA.csv'
data.to_csv(csv_EQ)

excel_file = 'total.xlsx'
data = pd.read_excel(excel_file)

csv_EQ = 'total.csv'
data.to_csv(csv_EQ)



data_Path = 'CSV_MA.csv'
ma_data = pd.read_csv(data_Path)

ma_data['Time'] = ma_data.apply(lambda row: convert_to_24hr(row['Time'], row['TZ']), axis=1)
ma_data.drop('TZ', axis=1, inplace=True)
print(ma_data)

ma_data.to_csv('CSV_MA.csv', index=False)


data_Path = 'total.csv'
ma_data = pd.read_csv(data_Path)

ma_data['Time'] = ma_data.apply(lambda row: convert_to_24hr(row['Time'], row['TZ']), axis=1)
ma_data.drop('TZ', axis=1, inplace=True)
print(ma_data)

ma_data.to_csv('total.csv', index=False)
# Drop intermediate column 'Time_24hr' to keep the DataFrame clean
#ma_data = ma_data.drop(columns=['Time_24hr'])

