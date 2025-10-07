from pybaseball import statcast
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os


data = statcast(start_dt = "2024-04-01", end_dt = "2024-09-30")

data.to_csv("statcast_2023.csv", index=False)

print(os.getcwd())

