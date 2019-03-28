We can make same examples where K-means and HAC have problems, especially with naive distance metric
-> K-means has a problem with initializiation and get stuck at local optima without going to check global optima
-> HAC if we have signal with noise the noise will dominate the signal by far

Moving to probabilistic setting:

Generative model, there is a hidden cluster label Z[n] is going to produce our data X[n], e.g. X is size of the car, Z is nationality of car
This is similar to classification setting, where you would build a classifier that given X[n] classify Z[n].

Z ~ Cat(pi), sampled from a categorical distribution with prior probabilities pi = [pi1, pi2, pi3, .. , pik]
X ~ N(Mz[n], Sz[n]), sampled from a gaussian distribution
X[n] ~ p(X[n] | Z[n], theta) where theta are the parameters of my generative model

if we want to predict p(z | x, Theta, pi) =~ p(x | z, Theta) * p(z | pi)
Same concept of bayesian generative models, but now we have this big unknown of not knowing the labels z

 > `Complete data loglikehood` is the probability p({X[n], Z[n]} | Theta)

We want to optimize Theta, the loss is
L(theta) = sum([log(p(X[i], Z[i] | Theta, pi) for i in range(n)])
	apply log properties to take out pi parameters
L(theta) = sum([log(p(X[i], Z[i] | Theta) + log(p(Z[n] | pi) for i in range(n)]) 
    write it down label by label
L(theta) = sum([ sum([  Z[i,j] * log(p(X[i] | Z[i] ,Theta) + Z[i, j] * log(pi[j]) for j in range(k)]) for i in range(n)]) 

from car example
pi[k] : prior knowledge about EU, US split
theta[k] : mean and variance for single EU, US distribution
X: is the weight of the car
Z: is the label of the car (from EU, from US)
assume no-covariance shift (all data points come from the same distribution)

There are two unknowns

1)  local {Z[n,k]}
2)  global {Theta, Pi}
solving for Z[n,k] is hard since we have a lot of variables, the surface is non convex

Approaches:

>  `Max-Max` is k-means in a probabilistic setting

1. given Theta, Pi -> set Z[n] to best cluster
2. given {Z[n]}, set Theta and Pi (just like classification)

Another approach, I might not want to commit to change my labeling when the cluster probability changes, since the two clusters could overlap.
(The probability changes when the two gaussian bells representing true clusters distribution intersect)

> `Expectation-Max` (E.M)

Let q[n, k] let be my distribution on Z[n]
e.g
Z[x] = [0,1,0,0] (for x in range(n))
q[x] = [0, 75, 25, 0] (for x in range(n)) # is the proabbility that a given x is from label k

from above
L(theta) = sum([ sum([  Z[i,j] * log(p(X[i] | Z[i] ,Theta) + Z[i, j] * log(pi[j]) for j in range(k)]) for i in range(n)]) 
l(theta) = sum([ sum([  z[i,j] * log(p(x[i] | theta[k]) + z[i, j] * log(pi[j]) for j in range(k)]) for i in range(n)])  #notation change

for a given x in n:
Expected[L(Theta)] =  Expected( sum([  z[x,j] * log(p(X[x] | theta[k]) + z[x, j] * log(pi[j]) for j in range(k)]) 
and now we use q
Expected[L(Theta)] =  sum([  q[x, j]* log(p(X[x] | theta[k]) + q[x, j] * log(pi[j]) for j in range(k)]
but q[n, k] is easy given Theta and Pi

p(Z[n]|X[n],Theta,Pi) =~ p(X[n] | Z[n], Theta) * p(Z[n] | Pi)
q[n, k] =~ p(X[n] | Theta[k]) * pi[k]

where p(X[n] | Theta[k])  ~ N(x[n]; m[k],S[k])

> `E-Step` (above) we are computing the expectation of the posterior distribution of the random variable

then we have a M step where we maximize, we don't have real labels like in classification but we see some patterns and we try to expose them

Observation: if the variance of a cluster S[k] is really small, we go back to MaxMax
