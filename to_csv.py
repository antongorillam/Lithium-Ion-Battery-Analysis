import os
import argparse
import pandas as pd
import numpy as np

from typing import List
from scipy import io

parser = argparse.ArgumentParser(description='Lol')
parser.add_argument('-p','--paths', nargs='+', help='Paths to the .m files to be converted', required=True, type=str)
parser.add_argument('--save_path', help='The folder to save the processed .csv file', default=None, type=str)
parser.add_argument('--file_name', help='The file name of the new .csv file', default=None, type=str)
parser.add_argument('--rul', help='The SoH where we deem the end-of-life of the battery (0-100)', default=None, type=int) # Refractor this
parser.add_argument('-q', '--query', nargs='+', help='', default=None, type=str)

def create_df(paths: List[str]) -> pd.DataFrame:
    """
    Read in a list of MATLAB files and create a Pandas DataFrame.

    Args:
        paths (List[str]): A list of file paths to read.

    Returns:
        pd.DataFrame: A DataFrame containing the data from all the files.
    """

    dfs = []
    for path in paths:
        # Obtain the name of the battery as a string
        file_name = os.path.basename(path)
        battery_name = os.path.splitext(file_name)[0]

        # Read .mat file as a dataframe
        mat = io.loadmat(path, simplify_cells=True)
        df = pd.DataFrame(mat['data']['step'])
        
        # Add the battery name as a column. For ex. "RW8"
        df['dataset'] = battery_name 
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def create_features(unprocessed_df: pd.DataFrame, rul: int) -> pd.DataFrame:
    """
    This function takes an unprocessed DataFrame and returns a processed DataFrame with additional features.
    
    Args:
        unprocessed_df (pd.DataFrame): The unprocessed DataFrame to be processed.
        rul: ---
        query: ---
    
    Returns:
        pd.DataFrame: The processed DataFrame with additional features.
    """
    
    format_string = "%d-%b-%Y %H:%M:%S"

    # Extract reference discharge data and compute capacity
    df_ref = unprocessed_df.query('comment=="reference discharge"')
    df_ref['df_ref '] = pd.to_datetime(df_ref['date'], format=format_string, errors='coerce')
    df_ref['capacity (Ah)'] = [np.trapz(i, t) / 3600 for i, t in zip(df_ref['current'], df_ref['relativeTime'])] # compute the capacity in Ah
    df_ref['gt'] = True

    # Process unprocessed_df
    unprocessed_df['dateTime'] = pd.to_datetime(unprocessed_df['date'], format=format_string, errors='coerce')
    unprocessed_df['date'] = unprocessed_df['dateTime'].apply(lambda x: x.date()) 
    unprocessed_df['time'] = unprocessed_df['time'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['voltage'] = unprocessed_df['voltage'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['current'] = unprocessed_df['current'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
    unprocessed_df['timeRange'] = unprocessed_df['relativeTime'].apply(lambda x: x[-1] if isinstance(x, np.ndarray) else x)
    unprocessed_df['avgTemperature'] = unprocessed_df['temperature'].apply(lambda x: np.mean(x))
    unprocessed_df['avgCurrent'] = unprocessed_df['current'].apply(lambda x: np.mean(x))
    unprocessed_df['avgVoltage'] = unprocessed_df['voltage'].apply(lambda x: np.mean(x))
    unprocessed_df['avgResistance'] = unprocessed_df['avgCurrent'] / unprocessed_df['avgVoltage']
    unprocessed_df['startVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[0])
    unprocessed_df['terminalVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[-1])
    unprocessed_df['deltaVoltage'] = unprocessed_df['terminalVoltage'] - unprocessed_df['startVoltage']
    unprocessed_df['rateOfVoltage'] = (unprocessed_df['deltaVoltage'] / unprocessed_df['timeRange']) / 3600 
    unprocessed_df = unprocessed_df.join(df_ref[['capacity (Ah)', 'gt']], how='left')
    unprocessed_df['gt'] = unprocessed_df['gt'].notna()
    unprocessed_df['endTime'] = unprocessed_df['time'].apply(lambda x: x[-1]) / 3600
    unprocessed_df['cycle'] = unprocessed_df.index

    # Compute SoH
    unprocessed_df['soh'] = unprocessed_df['capacity (Ah)'] / unprocessed_df['capacity (Ah)'].dropna().iloc[0] * 100 # Compute the SoH

    # Compute RUL
    eol_row = unprocessed_df.query('gt==True').iloc[
        (unprocessed_df.query('gt==True')['soh'] - rul).abs().argsort()[:1]]

    eol_cycle = eol_row.cycle.values[0]
    unprocessed_df['rul'] = unprocessed_df.cycle.apply(lambda x: eol_cycle - x)

    # Add daily average temperature
    mean_df = unprocessed_df.groupby(['date'])[['avgTemperature']].mean().reset_index()
    unprocessed_df = unprocessed_df.merge(mean_df, on=['date'], suffixes=('Cycle', 'Daily'))

    # Select relevant columns for processed DataFrame
    processed_df = unprocessed_df[[
            'comment',
            'type',
            'dataset',
            'dateTime',
            'endTime',
            'timeRange',
            'avgTemperatureCycle',
            'avgTemperatureDaily', 
            'avgVoltage',
            'avgCurrent',
            'avgResistance',
            'startVoltage',
            'terminalVoltage',
            'deltaVoltage',
            'rateOfVoltage',
            'gt',
            'cycle',
            'capacity (Ah)',
            'soh',
            'rul',
            ]]
    
    return processed_df

def save_csv(df: pd.DataFrame, save_path: str, file_name: str) -> None:
    """
    This function saves a Pandas DataFrame as a CSV file with the specified file name at the specified file path.
    If file_name is None, the function gets the unique names of the columns of df['dataset'] and appends them into a string,
    separated with "_" as the file_name.
    
    args:
        save_path (str): The file path where the CSV file will be saved.
        file_name (str): The name of the CSV file. If None, the unique names of the columns of df['dataset'] will be used.
        df (pd.DataFrame): The Pandas DataFrame to be saved as a CSV file.
    
    Returns:
        None
    """
    # If file_name is None, get the unique names of the columns of df['data'] and append them into a string
    if file_name is None:
        file_name = '_'.join(df['dataset'].unique())
    
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
        filtered_df = df.query('type in @query')
        return filtered_df.reset_index()
    else:
        return df

def main():
    args=parser.parse_args()
    
    unprocessed_df = create_df(args.paths)
    unprocessed_df = filter_df(unprocessed_df, args.query)
    processed_df = create_features(unprocessed_df, args.rul)
    save_csv(df=processed_df,
             save_path=args.save_path,
             file_name=args.file_name
             )

if __name__ == '__main__':
    main()