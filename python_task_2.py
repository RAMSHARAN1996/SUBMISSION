#question 1

import pandas as pd

def calculate_distance_matrix(dataset_path):
    # Read the dataset into a DataFrame
    df = pd.read_csv(dataset_path)

    # Create a DataFrame with source and destination columns as indices
    distance_matrix = pd.DataFrame(index=df['source'].unique(), columns=df['destination'].unique())

    # Fill the matrix with cumulative distances
    for _, row in df.iterrows():
        source, destination, distance = row['source'], row['destination'], row['distance']
        current_distance = distance_matrix.loc[source, destination]

        # Update the matrix with cumulative distances
        if pd.notna(current_distance):
            distance_matrix.loc[source, destination] += distance
            distance_matrix.loc[destination, source] += distance
        else:
            distance_matrix.loc[source, destination] = distance
            distance_matrix.loc[destination, source] = distance

    # Fill diagonal values with 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    return distance_matrix

dataset_path = 'dataset-3.csv'
result_distance_matrix = calculate_distance_matrix(dataset_path)
print(result_distance_matrix)

#question 2

import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Reset the index to include 'id_start' as a column
    distance_matrix = distance_matrix.reset_index()

    # Melt the DataFrame to convert it to long format
    melted_distance = pd.melt(distance_matrix, id_vars=['index'], var_name='id_end', value_name='distance')

    # Rename the columns to match the desired output
    melted_distance.columns = ['id_start', 'id_end', 'distance']

    # Filter out rows where 'id_start' is equal to 'id_end'
    melted_distance = melted_distance[melted_distance['id_start'] != melted_distance['id_end']]

    # Reset the index and drop the old index column
    melted_distance.reset_index(drop=True, inplace=True)

    return melted_distance

distance_matrix_q1 = calculate_distance_matrix('dataset-3.csv')  # Assuming you have the function from Question 1
result_unrolled = unroll_distance_matrix(distance_matrix_q1)
print(result_unrolled)

#question 3

import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter the DataFrame based on the reference value
    reference_df = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_avg_distance - (0.1 * reference_avg_distance)
    upper_bound = reference_avg_distance + (0.1 * reference_avg_distance)

    # Filter values within the 10% threshold and sort the result
    result_values = (
        df.groupby('id_start')
        .filter(lambda group: lower_bound <= group['distance'].mean() <= upper_bound)
        .sort_values('id_start')
        .drop_duplicates('id_start')
        ['id_start']
        .tolist()
    )

    return result_values

unrolled_df_q3 = unroll_distance_matrix(distance_matrix_q1)  # Assuming you have the function from Question 3
reference_value = 1  # Replace with the desired reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df_q3, reference_value)
print(result_ids_within_threshold)

#question 4

import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

unrolled_df_q3 = unroll_distance_matrix(distance_matrix_q1)  # Assuming you have the function from Question 3
result_df_with_toll_rates = calculate_toll_rate(unrolled_df_q3)
print(result_df_with_toll_rates)

#question 5

import pandas as pd
from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0)),
        (time(10, 0, 0), time(18, 0, 0)),
        (time(18, 0, 0), time(23, 59, 59))
    ]
    
    weekend_time_ranges = [
        (time(0, 0, 0), time(23, 59, 59))
    ]

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_datetime'].dt.day_name()
    df['start_time'] = df['start_datetime'].dt.time
    df['end_day'] = df['end_datetime'].dt.day_name()
    df['end_time'] = df['end_datetime'].dt.time

    # Calculate time-based toll rates
    for idx, row in df.iterrows():
        start_time, end_time = row['start_time'], row['end_time']

        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            discount_factors = weekday_discount_factors
            time_ranges = weekday_time_ranges
        else:
            discount_factors = [weekend_discount_factor]
            time_ranges = weekend_time_ranges

        toll_rate = 0

        for i, (start_range, end_range) in enumerate(time_ranges):
            if start_time <= start_range and end_time >= end_range:
                toll_rate = row['distance'] * discount_factors[i]
                break

        df.at[idx, 'toll_rate'] = toll_rate

    return df
  \
unrolled_df_q3 = unroll_distance_matrix(distance_matrix_q1)  # Assuming you have the function from Question 3
unrolled_df_q3['start_datetime'] = pd.to_datetime(unrolled_df_q3['start_datetime'])
unrolled_df_q3['end_datetime'] = pd.to_datetime(unrolled_df_q3['end_datetime'])
result_df_with_time_based_toll_rates = calculate_time_based_toll_rates(unrolled_df_q3)
print(result_df_with_time_based_toll_rates)


