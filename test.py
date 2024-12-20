import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('total_csv.csv')

new_col = ['DateTime', 'min','max','avg']

df = df[new_col]
print(df)
df.to_csv('total_csv.csv')