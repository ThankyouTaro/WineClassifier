import pandas as pd
rate_sum = {}
rate_num = {}

df = pd.read_csv('XWines_Full_21M_ratings.csv')

for index, row in df.iterrows():
    if row['WineID'] in rate_sum:
        rate_sum[row['WineID']] += int(row['Rating'])
        rate_num[row['WineID']] += 1
    else:
        rate_sum[row['WineID']] = int(row['Rating'])
        rate_num[row['WineID']] = 1

wine_id = []
wine_rate = []
for key in rate_sum:
    wine_id.append(key)

    wine_rate.append(round(rate_sum[key]/rate_num[key]))

output_dict = {'wine_id': wine_id, 'wine_rate': wine_rate}

df_out = pd.DataFrame(output_dict)
df_out.to_csv('test.csv')
