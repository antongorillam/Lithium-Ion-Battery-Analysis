import os
import argparse
import warnings
import pandas as pd
import numpy as np
import json

from tqdm import tqdm # for progress bar
from typing import List
from scipy import io

JSON_PATH = "anomaly_cycles.json"

# Ignore the specific warning
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Convert .mat files to .csv files and extract features.')
parser.add_argument('-p','--paths', nargs='+', help='Paths to the .m files to be converted', required=True, type=str)
parser.add_argument('--save_path', help='The folder to save the processed .csv file', default=None, type=str)
parser.add_argument('--file_name', help='The file name of the new .csv file', default=None, type=str)
parser.add_argument('-q', '--query', nargs='+', help='', default=None, type=str)
parser.add_argument('--make_monotonic', help='Whether to make the soh column monotonic', action='store_true')
parser.add_argument('--interpolation_method', help='The method to be used for interpolation', default='linear', type=str)

def create_df(paths: List[str]) -> List[pd.DataFrame]:
    """
    Read in a list of MATLAB files and create a Pandas DataFrame.

    Args:
        paths (List[str]): A list of file paths to read.

    Returns:
        pd.DataFrame: A DataFrame containing the data from all the files.

    Example:
        to_csv.py -p battery\ datasets/1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat --save_path processed_datasets -q D
    """

    dfs = []
    for path in tqdm(paths, ncols=75, desc=f'Reading battery'):
        # Obtain the name of the battery as a string
        file_name = os.path.basename(path)
        battery_name = os.path.splitext(file_name)[0]

        # Read .mat file as a dataframe
        mat = io.loadmat(path, simplify_cells=True)
        df = pd.DataFrame(mat['data']['step'])
        
        # Add the battery name as a column. For ex. "RW8"
        df['battery_name'] = battery_name 
        dfs.append(df)

    return dfs

def extrac_statistical_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function takes a DataFrame and a column name and extracts the statistical features of the column.
    These statistical features include the average, variance, maximum, minimum, kurtosis and skewness.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        column (str): The column name to extract the statistical features from.

    Returns:
        pd.DataFrame: The DataFrame with the statistical features added.
    """
    upper_column = column[0].upper() + column[1:]
    df[f'avg{upper_column}'] = df[column].apply(lambda x: np.mean(x))
    df[f'var{upper_column}'] = df[column].apply(lambda x: np.var(x))
    df[f'max{upper_column}'] = df[column].apply(lambda x: np.max(x))
    df[f'min{upper_column}'] = df[column].apply(lambda x: np.min(x))
    df[f'kurtosis{upper_column}'] = df[column].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    df[f'skewness{upper_column}'] = df[column].apply(lambda x: pd.Series(x).skew(skipna=True))
    # Drop nan for these columns
    return df.dropna(subset=[f'avg{upper_column}', f'var{upper_column}', f'max{upper_column}', f'min{upper_column}', f'kurtosis{upper_column}', f'skewness{upper_column}'])


def create_features(unprocessed_df: pd, query: list[str]) -> pd.DataFrame:
    """
    This function takes an unprocessed DataFrame and returns a processed DataFrame with additional features.
    
    Args:
        unprocessed_df (pd.DataFrame): The unprocessed DataFrame to be processed.
        query: ---
    
    Returns:
        pd.DataFrame: The processed DataFrame with additional features.
    """
    
    format_string = "%d-%b-%Y %H:%M:%S"

    # Extract reference discharge data and compute capacity
    df_ref = unprocessed_df.query('comment=="reference discharge"')
    df_ref['df_ref '] = pd.to_datetime(df_ref['date'], format=format_string, errors='coerce')
    df_ref['capacity (Ah)'] = 0.0
    df_ref['capacity (Ah)'] = [np.trapz(i, t) / 3600 for i, t in zip(df_ref['current'], df_ref['relativeTime'])] # compute the capacity in Ah

    # If the discharge cycle is not suppose to be here
    if 'D' not in query:
        df_ref_temp = unprocessed_df.query('comment=="reference charge"')
        df_ref_temp['capacity (Ah)'] = np.nan
        # loop over df_ref and df_ref_temp (which are the same lenght)
        # and replace df_ref_temp['capacity'] woth df_ref['capacity']
        for i in range(len(df_ref)):
            df_ref_temp['capacity (Ah)'].iloc[i] = df_ref['capacity (Ah)'].iloc[i] 

        # Assert that df_ref_temp['capacity (Ah)'] does not contain nan values
        assert not df_ref_temp['capacity (Ah)'].isna().any(), 'df_ref_temp["capacity (Ah)"] contains nan values'
        df_ref = df_ref_temp
        # Remove the type D from the unprocessed_df
        unprocessed_df = unprocessed_df.query('type!="D"')

    df_ref['gt'] = True

    # Process unprocessed_df
    unprocessed_df['dateTime'] = pd.to_datetime(unprocessed_df['date'], format=format_string, errors='coerce')
    unprocessed_df['date'] = unprocessed_df['dateTime'].apply(lambda x: x.date()) 
    unprocessed_df['current'] = unprocessed_df['current'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['time'] = unprocessed_df['time'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['timeRange'] = unprocessed_df['relativeTime'].apply(lambda x: x[-1] if isinstance(x, np.ndarray) else x)
    unprocessed_df['voltage'] = unprocessed_df['voltage'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['resistance'] = unprocessed_df['voltage'] / unprocessed_df['current']
    unprocessed_df['resistance'] = unprocessed_df['resistance'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))

    unprocessed_df = extrac_statistical_features(unprocessed_df, 'temperature')
    unprocessed_df = extrac_statistical_features(unprocessed_df, 'current')
    unprocessed_df = extrac_statistical_features(unprocessed_df, 'voltage')
    unprocessed_df = extrac_statistical_features(unprocessed_df, 'resistance')

    unprocessed_df['startVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[0])
    unprocessed_df['terminalVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[-1])
    unprocessed_df['deltaVoltage'] = unprocessed_df['terminalVoltage'] - unprocessed_df['startVoltage']

    # Compute the incremental change of the voltage vector
    # If the length of the voltage vector is 1, then the voltageRate is 0, no change is assumed
    unprocessed_df['voltageRate'] = unprocessed_df['voltage'].apply(lambda x: np.diff(x) if len(x) > 1 else np.array([0]))
    unprocessed_df = extrac_statistical_features(unprocessed_df, 'voltageRate')


    unprocessed_df = unprocessed_df.join(df_ref[['capacity (Ah)', 'gt']], how='left')
    unprocessed_df['gt'] = unprocessed_df['gt'].notna()
    unprocessed_df['endTime'] = unprocessed_df['time'].apply(lambda x: x[-1]) / 3600

    # Compute SoH
    unprocessed_df['soh'] = np.nan
    unprocessed_df['soh'] = unprocessed_df['capacity (Ah)'] / unprocessed_df['capacity (Ah)'].dropna().iloc[0] * 100 # Compute the SoH

    # Reset the index of the DataFrame
    unprocessed_df = unprocessed_df.reset_index(drop=True)
    unprocessed_df['cycle'] = unprocessed_df.index
    # Select relevant columns for processed DataFrame
    processed_df = unprocessed_df[[
            'comment',
            'type',
            'battery_name',
            'dateTime',
            'endTime',
            'timeRange',
            'avgTemperature',
            'varTemperature',
            'maxTemperature',
            'minTemperature',
            'kurtosisTemperature',
            'skewnessTemperature',
            'avgVoltage',
            'varVoltage',
            'maxVoltage',
            'minVoltage',
            'kurtosisVoltage',
            'skewnessVoltage',
            'avgCurrent',
            'varCurrent',
            'maxCurrent',
            'minCurrent',
            'kurtosisCurrent',
            'skewnessCurrent',
            'avgVoltageRate',
            'varVoltageRate',
            'maxVoltageRate',
            'minVoltageRate',
            'kurtosisVoltageRate',
            'skewnessVoltageRate',
            'avgResistance',
            'varResistance',
            'maxResistance',
            'minResistance',
            'kurtosisResistance',
            'skewnessResistance',
            'startVoltage',
            'terminalVoltage',
            'deltaVoltage',
            'gt',
            'cycle',
            'capacity (Ah)',
            'soh',
            ]]

    # Assert that the cycles are continuous
    assert np.all(np.diff(processed_df['cycle']) == 1), 'Cycles are not continuous'

    return processed_df

def to_monotonic_dec(serie: pd.Series):
    """
    This function takes a Pandas Series and returns a Pandas Series that is monotonic decreasing.
    args:
        serie (pd.Series): The Pandas Series to be made monotonic decreasing.
    returns:
        pd.Series: The monotonic decreasing Pandas Series.
    """
    return serie[serie <= serie.cummin()]

def interpolate_df(df: pd.DataFrame, method='pchip', make_monotonic=False) -> pd.DataFrame:
    """
    This function takes a DataFrame and interpolates the soh column for each unique battery.
    args:
        df (pd.DataFrame): The DataFrame to be interpolated.
        method (str): The method to be used for interpolation. Default is 'linear'.
        make_monotonic (bool): Whether to make the soh column monotonic. Default is False.
    returns:
        pd.DataFrame: The interpolated DataFrame.
    """
    # Assert that there is only one type of battery in the set
    assert len(df['battery_name'].unique()) == 1, 'There is more than one type of battery in the set'
    
    df_temp = df.query('gt==True')

    # A bit hard-coded for now, but this removes some anomaly cycles
    df_temp = remove_anomaly_cycles(df_temp)
    
    # Make soh column monotonic decreasing
    if make_monotonic:
        df_temp['soh'] = to_monotonic_dec(df_temp['soh'])
        # Replace the soh column in the original dataframe
        df['soh_interpolated'] = df_temp['soh']

    df['soh_interpolated'] = df['soh_interpolated'].interpolate(method=method, limit_area='inside')
    df = df.dropna(subset=['soh_interpolated']) # Drop nan values
    return df

def save_csv(df: pd.DataFrame, save_path: str, file_name: str) -> None:
    """
    This function saves a Pandas DataFrame as a CSV file with the specified file name at the specified file path.
    If file_name is None, the freunction gets the unique names of the columns of df['battery_name'] and appends them into a string,
    separated with "_" as the file_name.
    
    args:
        save_path (str): The file path where the CSV file will be saved.
        file_name (str): The name of the CSV file. If None, the unique names of the columns of df['battery_name'] will be used.
        df (pd.DataFrame): The Pandas DataFrame to be saved as a CSV file.
    
    Returns:
        None
    """
    # If file_name is None, get the unique names of the columns of df['data'] and append them into a string
    if file_name is None:
        file_name = '_'.join(df['battery_name'].unique())
    
    # Combine save_path and file_name to create full file path
    if save_path is None:
        full_path = file_name + '.csv'
    else:    
        full_path = save_path + '/' + file_name + '.csv'
    
    # Save DataFrame as CSV file
    df.to_csv(full_path, index=False)
    
    print(f'File sucessfully saved to "{full_path}"')
    return None

def filter_df(df: pd.DataFrame, query: List[str]) -> pd.DataFrame:
    """
    Filter the DataFrame based on a list of strings.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        query (List[str]): The list of strings to filter the DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    """
    # Filter the df with certain types
    if query is not None:
        query = query if 'D' in query else query + ['D'] # Add 'D' to the query if it is not there
        filtered_df = df.query('type in @query')
        return filtered_df.reset_index()
    else:
        return df

def remove_anomaly_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame and removes the cycles that are marked as anomaly in the json file.
    args:
        df (pd.DataFrame): The DataFrame to be filtered.

    returns:
        pd.DataFrame: The filtered DataFrame.
    """
    with open(JSON_PATH, 'r') as f:
        # Read the json file
        anomaly_cycles = json.load(f)
    
    current_battery = df['battery_name'].unique()[0]
    anomaly_cycles = anomaly_cycles[current_battery]

    # Exclude anomaly cycles during interpolation
    df = df[~df['cycle'].isin(anomaly_cycles)]
    return df

def main():

    args=parser.parse_args()
    
    # Create a list of unprocessed DataFrames
    unprocessed_dfs = create_df(args.paths)
    
    # Filter the DataFrames
    unprocessed_dfs = [filter_df(df, args.query) for df in tqdm(unprocessed_dfs, desc='Filtering data', ncols=75)]
    
    # Process the DataFrames e.g. extract features
    processed_dfs = [create_features(df, args.query) for df in tqdm(unprocessed_dfs, desc='Processing data', ncols=75)]
    
    # Interpolate the SoH column
    processed_dfs = [interpolate_df(df, make_monotonic=args.make_monotonic, method=args.interpolation_method) for df in tqdm(processed_dfs, desc='Interpolating SoH', ncols=75)]
    
    # Combine the processed DataFrames into one DataFrame
    processed_df = pd.concat(processed_dfs, ignore_index=False)
    
    # Save the processed DataFrame as a CSV file
    save_csv(   df=processed_df,
                save_path=args.save_path,
                file_name=args.file_name)

    tqdm.write(f'Data saved at {args.save_path}')

if __name__ == '__main__':
    main()