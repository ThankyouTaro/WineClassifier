import pandas as pd

df = pd.read_csv('Wine-Attribute-Ratings.csv')
to_delete = []
count = -1
for index, row in df.iterrows():
    count += 1
    if not str(row['region']).isdigit():
        to_delete.append(count)
        continue
    if not str(row['winery']).isdigit():
        to_delete.append(count)
        continue

print(to_delete)
new_df = df.drop(to_delete)
new_df.to_csv('WineData.csv', index=False)