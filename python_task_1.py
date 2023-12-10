import pandas as pd

def generate_car_matrix(dataset_path):
    # Read the dataset into a DataFrame
    df = pd.read_csv(dataset_path)

    # Pivot the DataFrame to create the desired matrix
    result_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    result_df.values[[range(len(result_df))]*2] = 0

    return result_df

dataset_path = 'dataset-1.csv'
result_matrix = generate_car_matrix(dataset_path)
print(result_matrix)

#question 2

import pandas as pd

def get_type_count(df):
    # Add a new categorical column 'car_type' based on 'car' values
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Count the occurrences of each car type
    type_counts = df['car_type'].value_counts()

    return type_counts


dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
result_counts = get_type_count(df)
print(result_counts)

#question 3

import pandas as pd

def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

dataset_path = 'dataset-1.csv'
df = pd.read_csv(dataset_path)
result_indexes = get_bus_indexes(df)
print(result_indexes)

#question 5

import pandas as pd

def multiply_matrix(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original DataFrame
    modified_df = input_df.copy()

    # Apply the multiplication logic to each element in the DataFrame
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

 (assuming result_matrix from Question 1)
result_matrix = generate_car_matrix('dataset-1.csv')
modified_result = multiply_matrix(result_matrix)
print(modified_result)

#question 6

import pandas as pd

def check_time_completeness(df):
    # Combine 'startDay' and 'startTime' to create a 'start_datetime' column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' to create an 'end_datetime' column
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Check if each (id, id_2) pair has incorrect timestamps
    completeness_check = (
        df.groupby(['id', 'id_2'])
        .apply(lambda group: (
            (group['start_datetime'].min().time() != pd.Timestamp('00:00:00').time()) or
            (group['end_datetime'].max().time() != pd.Timestamp('23:59:59').time()) or
            (set(group['start_datetime'].dt.weekday.unique()) != set(range(7)))
        ))
    )

    return completeness_check


dataset_path = 'dataset-2.csv'
df = pd.read_csv(dataset_path)
result = check_time_completeness(df)
print(result)
