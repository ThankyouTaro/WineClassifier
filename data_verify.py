import pandas as pd

df = pd.read_csv('Wine-Ratings.csv')

for index, row in df.iterrows():
    if row['wine_rate'] < 1:
        print('Found it!')
        print(index)
        break
