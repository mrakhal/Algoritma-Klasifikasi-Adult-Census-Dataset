from pandas import read_csv, DataFrame
import pandas as pd
import sys
from pandas_profiling import ProfileReport

df = pd.read_csv("adult.data")
profile = ProfileReport(df, minimal=True)
profile.to_file(output_file="output.html")