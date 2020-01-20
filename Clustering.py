""" Generating data"""
#we keep on running this part until we get 4 ceparate clusters 
from sklearn.datasets.samples_generator import make_blobs
Data, y=make_blobs(n_samples=600,n_features=2,centers=4)
import matplotlib.pyplot as plt 
feature_1 = Data[:,0]
feature_2 = Data[:,1]
plt.scatter(feature_1, feature_2)

"""Elbow method """
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(Data)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

""" Silhouette score """
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import numpy as np 
n_clusters = np.arange(2,11) #2 to 10 clusters ( n-1 ) 
for cluster in n_clusters: 
    model = KMeans(cluster, random_state=42)
    preds = model.fit_predict(Data)  #Since we had 10 clusters, we have 10 labels in the output i.e. 0 to 9
    score = silhouette_score (Data, preds)
    print (cluster ," : " , score) 
#the maximnum score corsp to the best nb of clusters

#☻visualize distribution     
for cluster in n_clusters: 
    model = KMeans(cluster, random_state=42)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
   
    visualizer.fit(Data)
    visualizer.poof()         # Fit the data to the visualizer
    visualizer.show() 
 
""" a better method to visualize silhouette score method """
    
import matplotlib.cm as cm  #changes the default colormap
range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]  
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13, 8)
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels = clusterer.fit_predict(Data)
    silhouette_avg = silhouette_score(Data, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(Data, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

   

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(Data[:, 0], Data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()   
    
    
""" Hierarchy clustering """    
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(Data)
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel(' Dataset')
plt.ylabel('n_clusters')
dendrogram(Z)
plt.show()

""" DBSCAN"""
#epsilon defines the maximum distance between two points from same class
#min_samples: The minimum number of neighbors a given point should have 
              #in order to be classified as a core point.
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

#finding the right epsilon 
neigh = NearestNeighbors(n_neighbors=2) #search for 2 closest neighbors 
nbrs = neigh.fit(Data)
distances, indices = nbrs.kneighbors(Data)

distances = np.sort(distances, axis=0) #every row is sorted 
distances = distances[:,1]
plt.plot(distances)
#The optimal value for epsilon will be found at the point of maximum curvature.
# selecting 0.5 for eps and setting min_samples to 5.

model = DBSCAN(eps=0.5, min_samples=5)
model.fit(Data)

clusters = model.labels_

colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(Data[:,0], Data[:,1], c=vectorizer(clusters))
#the blue color points is noise 


