import pandas as pd
type_dict = {}
ela_dict = {}
grape_dict = {}
harmon_dict = {}
body_dict = {}
acid_dict = {}
code_dict = {}
ratings = {}

wine_id = []
wine_type = []
wine_ela = []
wine_grape = []
wine_harmon = []
wine_abv = []
wine_body = []
wine_acid = []
wine_code = []
wine_region = []
wine_winery = []
wine_rating = []

df_rating = pd.read_csv('Wine-Ratings.csv')
df = pd.read_csv('XWines_Full_100K_wines.csv', encoding='ansi')

for index, row in df_rating.iterrows():
    ratings[row['wine_id']] = row['wine_rate']

for index, row in df.iterrows():
    if not row['WineID'] in ratings:
        continue
    wine_id.append(row['WineID'])
    if not row['Type'] in type_dict:
        next_sum = len(type_dict)
        # print(next_sum)
        type_dict[row['Type']] = next_sum
    wine_type.append(type_dict[row['Type']])
    if not row['Elaborate'] in ela_dict:
        next_sum = len(ela_dict)
        ela_dict[row['Elaborate']] = next_sum
    wine_ela.append(ela_dict[row['Elaborate']])
    if not row['Grapes'] in grape_dict:
        next_sum = len(grape_dict)
        grape_dict[row['Elaborate']] = next_sum
    wine_grape.append(grape_dict[row['Elaborate']])
    if not row['Harmonize'] in harmon_dict:
        next_sum = len(harmon_dict)
        harmon_dict[row['Harmonize']] = next_sum
    wine_harmon.append(harmon_dict[row['Harmonize']])

    wine_abv.append(row['ABV'])
    if not row['Body'] in body_dict:
        next_sum = len(body_dict)
        body_dict[row['Body']] = next_sum
    wine_body.append(body_dict[row['Body']])
    if not row['Acidity'] in acid_dict:
        next_sum = len(acid_dict)
        acid_dict[row['Acidity']] = next_sum
    wine_acid.append(acid_dict[row['Acidity']])
    if not row['Code'] in code_dict:
        next_sum = len(code_dict)
        code_dict[row['Code']] = next_sum
    wine_code.append(code_dict[row['Code']])
    wine_region.append(row['RegionID'])
    wine_winery.append(row['WineryID'])
    wine_rating.append(ratings[row['WineID']])

output_dict = {'id': wine_id, 'type': wine_type, 'ela': wine_ela, 'grapes': wine_grape,
               'harmon': wine_harmon, 'abv': wine_abv, 'body': wine_body, 'acid': wine_acid,
               'code': wine_code, 'region': wine_region, 'winery': wine_winery, 'rating': wine_rating}

output_df = pd.DataFrame(output_dict)


output_df.to_csv('Wine-Attribute-Ratings.csv', index=False)

