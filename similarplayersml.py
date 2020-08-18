# This code explores the NBA players from 2013 - 2014 basketball season, 
# and uses # a machine learning algorithm called kMeans to group them in 
# clusters, this will # show which players are most similar
# https://medium.com/@randerson112358/nba-data-analysis-exploration-9293f311e0e8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

#load the data 
nba = pd.read_csv('nba-2013-2014.csv')# the nba_2013.csv data 
nba.head(7)# Print the first 7 rows of data or first 7 players

#Get the number of rows and columns (481 rows or players , and 31 columns containing data on the players)
print("NBA Shape: "+ str(nba.shape)) # (481, 31) 
print("NBA Mean: "+ str(nba.mean()))

print("NBA FG Mean: "+ str(nba.loc[:,"fg"].mean()))

# Data Visualizations
# sns.pairplot(nba[["ast", "fg", "trb"]])
# plt.show()

# correlation = nba[["ast", "fg", "trb"]].corr()
# sns.heatmap(correlation, annot=True)


good_columns = nba._get_numeric_data().dropna(axis=1)

#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and #appended to our wcss list.
wcss=[]
for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(good_columns)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# KMeans - training the model
kmeans_model = KMeans(n_clusters=10, random_state=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
print("Labels")
print(labels)

# PCA - Plot a chart of the different clusters
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

# print(plot_columns)

# Iterate and view which cluster each player is in
cluster_label = []
for index, row in nba.iterrows():
    player_cols = good_columns.loc[ nba['player'] == row['player'],: ]
    player_list = player_cols.values.tolist()
    cluster_label.append(int(kmeans_model.predict(player_list)))
    
nba['cluster_label'] = cluster_label

# print(nba.loc[ nba['cluster_label'] == 2,: ])

nba.to_csv('nba-with-cluster.csv')