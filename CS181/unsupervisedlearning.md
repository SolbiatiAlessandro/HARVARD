Unsupervised Learning/Representation Learning (NN):
{x1,x2..xn} -> provide "summary" of xn

Applications:
- compresssion
- data exploration/understanding
- organize data
- hypotesis verification

How to evaluate unsupervised learning task?
minimize some notion of reconstruction error
reconstruction_error = sum([d(x, decode(encode(x))) for x in X]) # very solid metric
# encode is summary

NON_PROBABLISTIC CLUSTERING
Example: summary -> cluster
encoding: mapping Xn -> Mk (cluster center)
reconstruction_error =  sum([euclidean_distance(x, get_cluster_center(x)) for x in X ])

a representation for get_cluster_center(x) is one hot encoding [0, 0, 1, 0]

==============
Alg#1: K-MEANS
Goal: sum([distance(x, m)] for x in X for m in M where is_in_cluster(x, m))
Solution: Blocked Coordinate Ascent
if there is two sets of unknown  (x and m), solve one given the other is easy

def K-Means(X, M):
	"""
    X, points
    M, clusters where len(M) == K
	"""
	for i, x in enumerate(X):
		# give labels to closest cluster
		# (update cluster_labels Z given cluster_number #M)
		give_label(x, argmin([distance(x, m) for m in M]))
	for k, m in enumerate(M):
		# reposition centroids (cluster center) in the middle of the current cluster
		# (update cluster_number #M given cluster_labels Z)
		M[k] = 1/count([cluster(x) == k for x in X]) * sum([x for x in X if cluster(x) == l])

properties:
	- monotonic improvement
    - easly parallelizable/distributed computing
	- how to choose k -> there is no notion of overfitting, since we don't have truth label or generailzation
	- non-convex
    - boundaries are linear (you could get away with non-linear kernels)

=================================================
Alg#2: hierarchical agglomerative clustering (HAC)
build a dendrogram one step at a time, grouping together data point rather than splitting out the dataset until everyone is in one big cluster
can give you some insight on the number of clusters

1) every cluster is a singleton cluster
2) merge closest clusters together
	a) for point 2) we need some notion of distance between two points (you can generalize to any kind of distance)
	b) need notion for a linkage := distance between cluster (min linkage, max linkage, ward linkage)


