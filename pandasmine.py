from datetime import datetime

# Example load_data
load_data = {'start': ['12:34:56', '09:10:11']}

# Converting the string to time object
starttime = datetime.strptime(load_data['start'][0], '%H:%M:%S').time()

print(starttime)