from config import DATAPATH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df=pd.read_csv(DATAPATH, nrows=1000)
print('safaff')
print(df.head(20))
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
df.sort_values(by='timestamp',ascending=True,inplace=True)
df=df.set_index('timestamp')
st_date=pd.to_datetime('2015-06-12 00:00:00')
end_date=pd.to_datetime('2015-06-13 00:00:00')
df_second=df.resample('5S').mean()
df_24_hrs=df_second[(df_second.index >= st_date) & (df_second.index <= end_date)]
df_24_hrs.loc[:, 'Acceleration'] = np.sqrt(df_24_hrs['x']**2 + df_24_hrs['y']**2 + df_24_hrs['z']**2)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_24_hrs[['x','y','z','Acceleration']])


optimal_k = 4

# Perform K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

#  add cluster labels  to  DataFrame
df_24_hrs['cluster'] = cluster_labels

print(df_24_hrs.head(5))

