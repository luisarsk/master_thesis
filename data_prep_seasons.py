# ----------Date preparation----------
# generate two datasets, one for sommer and one for winter

import pandas as pd
import numpy as np

variable = 'SPV'
height = '0000m'
sufix = ''

# path = f'data/daily/H_ERA5_ECMW_T639_{variable}_{height}_daily_subcountry{sufix}.csv'
path  = f'data/prepared/{variable}_daily_subcountry_prepared.csv'
df = pd.read_csv(path, comment="#", parse_dates=["Date"])
df = df.set_index("Date")

# df_sommer = df[df.index.month.isin([4,5,6,7,8,9])]
# df_winter = df[df.index.month.isin([10,11,12,1,2,3])]
df_JJA = df[df.index.month.isin([6,7,8])]
df_DJF = df[df.index.month.isin([12,1,2])]

# path_save = f'data/daily/H_ERA5_ECMW_T639_{variable}_{height}_daily_subcountry{sufix}'
path_save = f'data/prepared/{variable}_daily_subcountry_prepared'
# df_sommer.reset_index().to_csv(f"{path_save}_sommer.csv", index=False)
# df_winter.reset_index().to_csv(f"{path_save}_winter.csv", index=False)
df_JJA.reset_index().to_csv(f"{path_save}_JJA.csv", index=False)
df_DJF.reset_index().to_csv(f"{path_save}_DJF.csv", index=False)

print('DONE!!!')