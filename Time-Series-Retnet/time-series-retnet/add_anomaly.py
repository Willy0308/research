import pandas as pd
import numpy as np

# Load the data
dataframe = pd.read_csv('../ausdata.csv')

# Ensure 'date' is a datetime type
dataframe['date'] = pd.to_datetime(dataframe['date'])

# Filter the date range
start_date = '2010-01-31'
end_date = '2010-12-31'
mask = (dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)
filtered_data = dataframe.loc[mask]

# Initialize a column for anomalies
dataframe['Anomaly'] = 0  # No anomalies by default

# Iterate over each day and choose a random time point to double the 'Load'
unique_days = filtered_data['date'].dt.date.unique()
for day in unique_days:
    day_mask = dataframe['date'].dt.date == day
    day_indices = dataframe[day_mask].index
    random_index = np.random.choice(day_indices)
    dataframe.loc[random_index, 'Load'] *= 2  # Double the Load value
    dataframe.loc[random_index, 'Anomaly'] = 1  # Mark as anomaly

# Save the modified dataframe
dataframe.to_csv('../modified_ausdata.csv', index=False)
print("Modified dataset saved as 'modified_ausdata.csv'.")
