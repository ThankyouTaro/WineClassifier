import os
import shutil
import pandas as pd


df = pd.read_csv('Wine-Ratings.csv')
diction = dict(zip(df['wine_id'], df['wine_rate']))
genpath = 'Images'
datanames = os.listdir(genpath)
count = -1
for i in datanames:

    count += 1
    name = i.split('.')[0]
    # print(name)
    if int(name) not in diction.keys():
        continue
    print(name)
    oldpath = genpath

    newpath = ''
    name_int = int(name)
    if count > len(datanames) * 0.95:
        newpath += 'test/' + str(diction[name_int])
    else:
        newpath += 'train/' + str(diction[name_int])
    src = os.path.join(oldpath, i)
    dst = os.path.join(newpath, i)
    shutil.move(src, dst)
