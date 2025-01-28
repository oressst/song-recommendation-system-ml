from scipy.spatial import distance
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/dataset.csv')
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")



# print("First 5 rows: ", df.head())
# print("Last 5 rows: ", df.tail())

# duplicated_rows = df.duplicated().sum()

# if duplicated_rows == 0:
#     print('There are 0 rows that are duplicated, which means each row in the DataFrame is unique.')
#     print('So that we do not need to continue processing duplicate lines')
# else:
#     print(f'There are {duplicated_rows} rows that are duplicated so we need to drop those {duplicated_rows} rows')
#     df = df.drop_duplicates()
#     print(f'After drop duplicated rows, there are {df.shape[0]} rows left')


# Display data types of all columns
# print("\nColumn data types:")
# print(df.dtypes)


# def open_object_dtype(s):
#     # print(f"Function called with series: {s.name}") 
#     dtypes = set()
#     dtypes.update(s.apply(type))
#     # print(f"Found types: {dtypes}")  
#     return dtypes

# obj_cols = df.select_dtypes(include='object').columns
# print(f"Object columns found: {obj_cols}")  

# # Store and print the result
# result = df[obj_cols].apply(open_object_dtype, axis=0).to_frame('Data Type')
# print("\nFinal result:")
# print(result)


# missing_values_per_row = df.isnull().sum(axis=1)
# count_per_missing_value = missing_values_per_row.value_counts().sort_index()

# # Print the results
# for missing, rows in count_per_missing_value.items():
#     print(f'{rows} row(s) have {missing} missing values')

# total_rows_with_missing_values = (df.isnull().any(axis=1)).sum()
# print(f'Total number of rows with missing values: {total_rows_with_missing_values}')


# numerical_cols = df[df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64')]]
# print(numerical_cols.shape)
# print(numerical_cols.sample(5))

# # numerical_cols.info()
# dist_numerical_cols = numerical_cols.describe().T[['min', 'max']]
# dist_numerical_cols['Missing Values'] = numerical_cols.isnull().sum()
# dist_numerical_cols['Missing Percentage'] = (numerical_cols.isnull().mean() * 100).round(2)
# # The number of -1 values in the 'key' column
# dist_numerical_cols.loc['key', 'Missing Values'] = (df['key'] == -1).sum()
# print(dist_numerical_cols)

# print(numerical_cols.describe())

# sns.set_style('darkgrid')
# sns.set_theme(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# numerical_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
# plt.tight_layout()
# plt.show()


categorical_cols = df[df.columns[(df.dtypes == 'object') | (df.dtypes == 'bool')]]
# print(categorical_cols.shape)
# print(categorical_cols.sample(5))

# print(categorical_cols.info())


# dist_categorical_cols = pd.DataFrame(
#     data = {
#         'Missing Values': categorical_cols.isnull().sum(),
#         'Missing Percentage': (categorical_cols.isnull().mean() * 100)
#     }
# )
# print(dist_categorical_cols)

# print(categorical_cols[categorical_cols.isnull().any(axis=1)])

# index_to_drop = df[categorical_cols.isnull().any(axis=1)].index
# df.drop(index_to_drop, inplace=True)

# print(f'Rows with missing values dropped. Updated DataFrame shape: {df.shape}')

# print(df.describe(include=['object', 'bool']))
# unique_values, value_counts = np.unique(categorical_cols['explicit'], return_counts=True)

# fig, ax = plt.subplots(figsize=(5, 5))

# # Explode the slice with explicit tracks for emphasis
# explode = [0, 0.1]  # Only "yes" (true) will be slightly exploded
# colors = ['#66b3ff','#99ff99']

# ax.pie(value_counts, labels=unique_values, autopct='%1.2f%%', startangle=90, colors=colors, explode=explode)
# ax.axis('equal')
# ax.set_title('Distribution of Explicit Tracks')
# plt.show()


# top_n = 10
# sns.set_style('darkgrid')
# sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})

# top_artists = df['artists'].value_counts().head(top_n)
# top_albums = df['album_name'].value_counts().head(top_n)
# top_tracks = df['track_name'].value_counts().head(top_n)
# top_genres = df['track_genre'].value_counts().head(top_n)

# # Disable FutureWarning
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=FutureWarning)

#     # Plotting
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

#     # Top N Artists
#     sns.barplot(x=top_artists.values, y=top_artists.index, palette="crest", ax=axes[0, 0], orient='h',  zorder=3, width=0.5)
#     axes[0, 0].set_title(f'Top {top_n} Artists')
#     axes[0, 0].set_xlabel('Frequency')
#     axes[0, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)
#     # Top N Albums
#     sns.barplot(x=top_albums.values, y=top_albums.index, palette="crest", ax=axes[0, 1], orient='h', zorder=3, width=0.5)
#     axes[0, 1].set_title(f'Top {top_n} Albums')
#     axes[0, 1].set_xlabel('Frequency')
#     axes[0, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     # Top N Tracks
#     sns.barplot(x=top_tracks.values, y=top_tracks.index, palette="crest", ax=axes[1, 0], orient='h', zorder=3, width=0.5)
#     axes[1, 0].set_title(f'Top {top_n} Tracks')
#     axes[1, 0].set_xlabel('Frequency')
#     axes[1, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     # Top N Genres
#     sns.barplot(x=top_genres.values, y=top_genres.index, palette="crest", ax=axes[1, 1], orient='h', zorder=3, width=0.5)
#     axes[1, 1].set_title(f'Top {top_n} Genres')
#     axes[1, 1].set_xlabel('Frequency')
#     axes[1, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     plt.tight_layout()
#     plt.show()



# boxplot for numerical columns
# sns.set_style('darkgrid')
# sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# columns = ['popularity', 'duration_ms', 'tempo', 'loudness', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
# fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
# for i, col in enumerate(columns):
#     sns.boxplot(y=col, data=numerical_cols, ax=axes[i//4, i%4])
#     axes[i//4, i%4].set_title(col)
# plt.tight_layout()
# plt.show()


# heatmap for correlation
# corr = numerical_cols.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# sns.set_style('white')
# sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# plt.figure(figsize=(15, 10))
# sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1,cmap='coolwarm')
# plt.show()


# df = df.drop(['time_signature', 'key'], axis=1)
# print(df.head(5))

# df.drop_duplicates(subset=['track_id'], inplace=True)
# print(df.head(5))

from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
import numpy as np

# Step 1: Data Preparation - as described in the document
df = df.drop(['time_signature', 'key'], axis=1)
df.drop_duplicates(subset=['track_id'], inplace=True)

# Step 2: Normalize the data using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
numerical_cols = df.select_dtypes(include=np.number).columns
data_norm = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), 
                        columns=numerical_cols, 
                        index=df['track_id'])

# Step 3: Get target track information
trackNameListened = "One That Got Away"
track_id = df[(df['track_name'] == trackNameListened)][['track_id']]
track_id = track_id.values[0][0]

# Step 4: Get the target track features
target_track = list(data_norm.loc[track_id])

# Step 5: Calculate Euclidean distances
data_result = pd.DataFrame()
data_result['euclidean'] = [distance.euclidean(list(obj), target_track) 
                           for _, obj in data_norm.iterrows()]
data_result['track_id'] = data_norm.index

# Optional: Sort results by distance to get recommendations
data_result = data_result.sort_values('euclidean')

data_rec = data_result.sort_values(by=['euclidean']).iloc[:6]

print(data_rec)

data_init = df.set_index(df.loc[:, 'track_id'])
track_list = pd.DataFrame()
for i in list(data_rec.loc[:, 'track_id']):
    if i in list(df.loc[:, 'track_id']):
        track_info = data_init.loc[[i], ['track_name', 'artists']]
        track_list = pd.concat([track_list, track_info], ignore_index=True)

print(track_list)


recomended = track_list.values.tolist()
print(f"""You've just listened:  \n \t - {recomended[0][0]} - {recomended[0][1]} 
Now you may listen : 
\n \t - '{recomended[1][0]} - {recomended[1][1]}'
Or any of:
\n \t - '{recomended[2][0]} - {recomended[2][1]}' 
\n \t - '{recomended[3][0]} - {recomended[3][1]}'
\n \t - '{recomended[4][0]} - {recomended[4][1]}'
\n \t - '{recomended[5][0]} - {recomended[5][1]}'  """)