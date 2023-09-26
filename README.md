# Lithium-Ion-Battery-Analysis

This github repo contains the code for the analysis of the lithium-ion battery data. The primary used data is :

- [NASA's Randomized Battery Usage](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) 

## Environment

The code uses `Anaconda 3` and `Python 3.10.11`. The dependencies can be found in the `environment.yml` file. To install the dependencies run the following command in the terminal:

```bash
conda env create -f environment.yml
```

## Run

### nasa_to_csv.py
The most relevant part of the code is `nasa_to_csv.py` which converts the NASA data from `.mat` to `.csv` format. This code does primarily these things:

- Filter out desired cycles (e.g. one can choose to only include charge cycles).
- Compute the SOH for each for the reference discharge cycle.
- Interpolate the SOH for each cycle (the interpolation disregard anomaly cycles which is found manually in `anomaly_cycles.json`)
- Extract features from each cycle (e.g. the mean, variance, etc. of the voltage, current, etc.).
- Save the data to `.csv` format.

The code also extracts features from each cycle. The code can be run using the following command:

```bash
python3 nasa_to_csv.py -p battery datasets/1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat battery datasets/1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat --save_path processed_datasets -q C --make_monotonic --interpolation_method linear
```

or alternatively, run debug on vscode after adding this to your `launch.json` file in `vscode`:
    
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: Module",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "-p", 
                    "battery datasets/1_Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat",
                    "battery datasets/1_Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat",
                    "battery datasets/1_Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat",
                    "battery datasets/1_Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat",
                    "battery datasets/2_Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW3.mat",
                    "battery datasets/2_Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW4.mat",
                    "battery datasets/2_Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW5.mat",
                    "battery datasets/2_Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW6.mat",
                    "battery datasets/3_Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1.mat",
                    "battery datasets/3_Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW2.mat",
                    "battery datasets/3_Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW7.mat",
                    "battery datasets/3_Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW8.mat",
                    "battery datasets/4_RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25.mat",
                    "battery datasets/4_RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26.mat",
                    "battery datasets/4_RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27.mat",
                    "battery datasets/4_RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28.mat",
                    "battery datasets/5_RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
                    "battery datasets/5_RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18.mat",
                    "battery datasets/5_RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
                    "battery datasets/5_RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
                    "battery datasets/6_RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21.mat",
                    "battery datasets/6_RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
                    "battery datasets/6_RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23.mat",
                    "battery datasets/6_RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24.mat",
                    "battery datasets/7_RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
                    "battery datasets/7_RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
                    "battery datasets/7_RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
                    "battery datasets/7_RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
                    "--save_path", "processed_datasets",
                "-q", "C",
                "--make_monotonic",
                "--interpolation_method", "linear",
            ],
            "justMyCode": true,

        }
    ]
}
```