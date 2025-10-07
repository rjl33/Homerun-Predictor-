import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

filepath = "2024_Statcast2.csv"

df = pd.read_csv(filepath)
useful_colmns = ['Release Speed', 'Batte1 Stance', 'Pitche1 Th1ow', 'Batter Advantage', 'Launch Speed', 'Launch Angle', 'Effective Speed', 'Spin rate', 'Pitch Number', 'Pitch Type']
df = df.dropna()
X = df[useful_colmns] #all parameters
y = df["Homerun?"]  #Target label

#Scale values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Convert torch tensors
X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)









