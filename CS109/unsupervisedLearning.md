more common to be in unsupervised than supervised
PCA (109A), multivariate data rotating around so that is captured in principal component -> goal: get low dimensionality
we use PCA as a visualization tool
identify clusters of observations that are homogeneus (similar to each other)

Cluster Analysis (this and next lecture):
- inter-observation distances
- partition-based clustering (k-means) -> specifiy number of clusters and then invoke algo
- hierarchical clustering -> merge data iteratevily and decide partitions afterward
- DBScan (?) on wednesday
- diagnostics and optimizing the number of clusters
- plots

Cluster: finding subgrups (clusters, subset of rows of X) in a multivariate data set where observation are similar.
Question: What does it mean observation similar?  construct the notion of interobservation distance
e.g. preprocess with CNN and then use the output of CNN as a distance metric, you can be creative here
you need to define distance metric
- Euclidean dist
- Manhattan dist
correlation based measures
- Pearson correlation distance
- Spearman correlation distance (with ranking, not to break with outliers)
you can also parametrize distance measures

scaling is important for distances

visualization to examine possible clusters: scatter plots between continuos variable

HIERARCHICAL CLUSTERING:

aggregation approaches: Ward's Method (centroid)
variance of observation between cluster
best reduction in within-cluster sum of squared distance.

how to choose optimal number of clusters:
--> Direct Methods

- Elbow method: 
T(k) = sum([within-cluster variation for cluster in clusters])
plot T(k) over number over clusters k: "elbow plot", see where the slope change significantly (elbow shape)

- AverageSilhouette method:
S(k) average of the silhouette across all observations for each k
S(k) = sum([silhouette(i) for i in observations])/len(observations)

where silhouette: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance.

-> Testing Methods

- Gap statistic:
is there a strong evidence that data clustered into K groups is significantly bettere than if they were generated at random?
Compare total within cluster expected sum of square distance.

(1) Cluster data with varying number of total clusters K, and T(k)
(2) Generate B reference data sets of size n, with the simulated values of variable j uniformly generated over the range of the observed variable x(j)
(3) For each generated dataset b = 1...B perform (1)
(4) Gap statistic(k) =  sum([log(T(k, b) for b in xrange(1, B)]) - log(T(k))
(5) perform a test on the statistic
(you can plot the Gap Statistic plot and apply the [gap statistic procedure])


DENSITY BASED CLUSTERING:
can find any shape of cluster (what if data are ring shaped)
identify observations that don't belong to any cluster
dose not require specify nums of clusters

e: radius of a neighborhood around an observation
minpts: min number of points in the neighborhood

distinguish points in Core points, Border points ans Noise points
pair of points can be Density reachable or density connected
cluster is defined as group of density reachable (?)

how to choose e? refer to paper "Ester et al."


(chernoff faces)


CLUSTERING:

scaling: have everything on a comparable scale: find a metrics that make sense for the data
http://web.stanford.edu/class/ee103/visualizations/kmeans/kmeans.html
what can go wrong with K means? if I am not initialized correctly  I can't really figure it out

crossvalidation: I don't have ground truth

exercize:
scaled (what do I mean by similar?)
add N-dimensional versor in a 2d mensional scatter plot of a PCA
