import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def load_trc_files(file_path):
    """
    Reads a OpenSim .trc marker file, skips the metadata headers,
    and returns a clean pandas DataFrame and its original frame rate.
    """   
    #read data rate from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_rate = float(lines[2].split()[0])