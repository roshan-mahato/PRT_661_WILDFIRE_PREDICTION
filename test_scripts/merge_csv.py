import pandas as pd

df1 = pd.read_csv('data\\weather_data.csv')
df2 = pd.read_csv('data/weather_data_new_laptop_test.csv')

print(f'weather_data row count: {len(df1)}')
print(f'weather_data_new_laptop_test row count: {len(df2)}')

merged_df = pd.concat([df1, df2], ignore_index=True)
print(f'merged row count: {len(merged_df)}')

merged_df.to_csv('data/all_weather_data.csv', index=False)