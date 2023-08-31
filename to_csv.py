import os
import argparse
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm # for progress bar
from typing import List
from scipy import io

# Ignore the specific warning
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Lol')
parser.add_argument('-p','--paths', nargs='+', help='Paths to the .m files to be converted', required=True, type=str)
parser.add_argument('--save_path', help='The folder to save the processed .csv file', default=None, type=str)
parser.add_argument('--file_name', help='The file name of the new .csv file', default=None, type=str)
parser.add_argument('-q', '--query', nargs='+', help='', default=None, type=str)

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

def create_features(unprocessed_df: pd.DataFrame) -> pd.DataFrame:
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

    unprocessed_df['avgTemperature'] = unprocessed_df['temperature'].apply(lambda x: np.mean(x))
    unprocessed_df['varTemperatureCycle'] = unprocessed_df['temperature'].apply(lambda x: np.var(x))
    unprocessed_df['maxTemperatureCycle'] = unprocessed_df['temperature'].apply(lambda x: np.max(x))
    unprocessed_df['minTemperatureCycle'] = unprocessed_df['temperature'].apply(lambda x: np.min(x))
    unprocessed_df['kurtosisTemperatureCycle'] = unprocessed_df['temperature'].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    unprocessed_df['skewnessTemperatureCycle'] = unprocessed_df['temperature'].apply(lambda x: pd.Series(x).skew(skipna=True))
    # Drop nan for the colums avgTemperature, varTemperature, maxTemperature, minTemperature, kurtosisTemperature, skewnessTemperature
    unprocessed_df = unprocessed_df.dropna(subset=['avgTemperature', 'varTemperatureCycle', 'maxTemperatureCycle', 'minTemperatureCycle', 'kurtosisTemperatureCycle', 'skewnessTemperatureCycle'])

    unprocessed_df['avgCurrent'] = unprocessed_df['current'].apply(lambda x: np.mean(x))
    unprocessed_df['varCurrent'] = unprocessed_df['current'].apply(lambda x: np.var(x))
    unprocessed_df['maxCurrent'] = unprocessed_df['current'].apply(lambda x: np.max(x))
    unprocessed_df['minCurrent'] = unprocessed_df['current'].apply(lambda x: np.min(x))
    unprocessed_df['kurtosisCurrent'] = unprocessed_df['current'].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    unprocessed_df['skewnessCurrent'] = unprocessed_df['current'].apply(lambda x: pd.Series(x).skew(skipna=True))
    unprocessed_df = unprocessed_df.dropna(subset=['avgCurrent', 'varCurrent', 'maxCurrent', 'minCurrent', 'kurtosisCurrent', 'skewnessCurrent'])

    unprocessed_df['avgVoltage'] = unprocessed_df['voltage'].apply(lambda x: np.mean(x))
    unprocessed_df['varVoltage'] = unprocessed_df['voltage'].apply(lambda x: np.var(x))
    unprocessed_df['maxVoltage'] = unprocessed_df['voltage'].apply(lambda x: np.max(x))
    unprocessed_df['minVoltage'] = unprocessed_df['voltage'].apply(lambda x: np.min(x))
    unprocessed_df['kurtosisVoltage'] = unprocessed_df['voltage'].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    unprocessed_df['skewnessVoltage'] = unprocessed_df['voltage'].apply(lambda x: pd.Series(x).skew(skipna=True))
    unprocessed_df = unprocessed_df.dropna(subset=['avgVoltage', 'varVoltage', 'maxVoltage', 'minVoltage', 'kurtosisVoltage', 'skewnessVoltage'])

    unprocessed_df['avgResistance'] = unprocessed_df['resistance'].apply(lambda x: np.mean(x))
    unprocessed_df['varResistance'] = unprocessed_df['resistance'].apply(lambda x: np.var(x))
    unprocessed_df['maxResistance'] = unprocessed_df['resistance'].apply(lambda x: np.max(x))
    unprocessed_df['minResistance'] = unprocessed_df['resistance'].apply(lambda x: np.min(x))
    unprocessed_df['kurtosisResistance'] = unprocessed_df['resistance'].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    unprocessed_df['skewnessResistance'] = unprocessed_df['resistance'].apply(lambda x: pd.Series(x).skew(skipna=True))
    unprocessed_df = unprocessed_df.dropna(subset=['avgResistance', 'varResistance', 'maxResistance', 'minResistance', 'kurtosisResistance', 'skewnessResistance'])

    unprocessed_df['startVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[0])
    unprocessed_df['terminalVoltage'] = unprocessed_df['voltage'].apply(lambda x: x[-1])
    unprocessed_df['deltaVoltage'] = unprocessed_df['terminalVoltage'] - unprocessed_df['startVoltage']

    # Compute the incremental change of the voltage vector
    # If the length of the voltage vector is 1, then the voltageRate is 0, no change is assumed
    unprocessed_df['voltageRate'] = unprocessed_df['voltage'].apply(lambda x: np.diff(x) if len(x) > 1 else np.array([0]))
    unprocessed_df['avgVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: np.mean(x))
    unprocessed_df['varVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: np.var(x))
    unprocessed_df['maxVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: np.max(x))
    unprocessed_df['minVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: np.min(x))
    unprocessed_df['kurtosisVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: pd.Series(x).kurtosis(skipna=True))
    unprocessed_df['skewnessVoltageRate'] = unprocessed_df['voltageRate'].apply(lambda x: pd.Series(x).skew(skipna=True))
    unprocessed_df = unprocessed_df.dropna(subset=['avgVoltageRate', 'varVoltageRate', 'maxVoltageRate', 'minVoltageRate', 'kurtosisVoltageRate', 'skewnessVoltageRate'])

    unprocessed_df['rateOfVoltageLEGACY'] = (unprocessed_df['deltaVoltage'] / unprocessed_df['timeRange']) / 3600 
    unprocessed_df = unprocessed_df.join(df_ref[['capacity (Ah)', 'gt']], how='left')
    unprocessed_df['gt'] = unprocessed_df['gt'].notna()
    unprocessed_df['endTime'] = unprocessed_df['time'].apply(lambda x: x[-1]) / 3600

    # Compute SoH
    unprocessed_df['soh'] = np.nan
    unprocessed_df['soh'] = unprocessed_df['capacity (Ah)'] / unprocessed_df['capacity (Ah)'].dropna().iloc[0] * 100 # Compute the SoH

    # Add daily average temperature
    mean_df = unprocessed_df.groupby(['date'])[['avgTemperature']].mean().reset_index()
    unprocessed_df = unprocessed_df.merge(mean_df, on=['date'], suffixes=('Cycle', 'Daily'))

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
            'avgTemperatureCycle',
            'avgTemperatureDaily',
            'varTemperatureCycle',
            'maxTemperatureCycle',
            'minTemperatureCycle',
            'kurtosisTemperatureCycle',
            'skewnessTemperatureCycle',
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
            'rateOfVoltageLEGACY',
            'gt',
            'cycle',
            'capacity (Ah)',
            'soh',
            ]]

    # Assert that the cycles are continuous
    assert np.all(np.diff(processed_df['cycle']) == 1), 'Cycles are not continuous'

    return processed_df

def save_csv(df: pd.DataFrame, save_path: str, file_name: str) -> None:
    """
    This function saves a Pandas DataFrame as a CSV file with the specified file name at the specified file path.
    If file_name is None, the function gets the unique names of the columns of df['battery_name'] and appends them into a string,
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
        filtered_df = df.query('type in @query')
        return filtered_df.reset_index()
    else:
        return df

def main():

    args=parser.parse_args()
    # Create a list of unprocessed DataFrames
    unprocessed_dfs = create_df(args.paths)
    # Filter the DataFrames
    unprocessed_dfs = [filter_df(df, args.query) for df in tqdm(unprocessed_dfs, desc='Filtering data', ncols=75)]
    # Process the DataFrames e.g. extract features
    processed_dfs = [create_features(df) for df in tqdm(unprocessed_dfs, desc='Processing data', ncols=75)]
    processed_df = pd.concat(processed_dfs, ignore_index=False)    
    # Save the processed DataFrame as a CSV file
    save_csv(   df=processed_df,
                save_path=args.save_path,
                file_name=args.file_name)
    tqdm.write(f'Data saved at {args.save_path}')

if __name__ == '__main__':
    main()