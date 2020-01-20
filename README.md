# Clustering_methods-

In this project I did 4 different clustering methods. 

# Elbow method : 
The Elbow method looks at the total WSS as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn’t improve much better the total WSS.
![alt text](https://www.datanovia.com/en/wp-content/uploads/dn-tutorials/004-cluster-validation/figures/015-determining-the-optimal-number-of-clusters-k-means-optimal-clusters-wss-silhouette-1.png)

# Silhouette score : 
Average silhouette method computes the average silhouette of observations for different values of k. The optimal number of clusters k is the one that maximize the average silhouette over a range of possible values for k
![alt text](https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_002.png )
![alt text](https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_003.png )
![alt text](https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_004.png)

#DBSCAN : 
DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.
![alt text](http://muthu.co/wp-content/uploads/2018/07/Snip20180707_105.png)

# Hierearchical clustering : 
Hierarchical clustering, also known as hierarchical cluster analysis, is an algorithm that groups similar objects into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other.

![alt tewt](https://python-graph-gallery.com/wp-content/uploads/400_Basic_Dendrogram.png)
