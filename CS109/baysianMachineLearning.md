Are you Bayesian?
================
1. A music expert claims he can identify whether a piece of music was written by Mozart versus Haydn.
In 10 trials, the expert identifies the composer correctly 10 times. What do you think about the next trial?
2. A drunken friend claims he can predict the result of coin flips.
In 10 trials, he answers correctly 10 times. What do you think about the next trial?
3. A coffee connoisseur claims to be able to tell the difference between whether cream is pured in coffee first, or vice versa.
In 10 trials, the person answers correctly 10 times. What do you think about the next trial?

If you drew different conclusions in all three situations, then you are likely already Bayesian.

> Bayesian statistic is a `learning paradigm`. You cannot ignore knowledge you currently have in summarizing conclusions based on data.

The differnce between frequentist and Bayesian statistics is in the definition of probability.
Example: flip a fair coin, the probability the coin lands heads is 0.5. 
What does `the probability is 0.5` mean?
* frequentist: 0.5 of the time the coin will land head
* bayesian: there is an equal probability of heads and tails due to the symmetry of the coin

> Frequentist: probability of an event is its long-run frequency of occurence (classical probability)
> Bayesian: probability of an event is one's degree of belief that the event will occur (subjective probability)

Implications:
============
* frequentist: can only assign probabilities to future data or potentially observable quantities
* bayesian: can consider probabilities of values with unknown parameters

Variation of coin flipping:
- Yesterday, at 17:00, at home, I flipped a coin.
* frequentist: since what I am asking you about a specific event probability is either 0 or 1
* bayesian: 0.5

Example: confidence intervals
============================
I have a random sample Y = [y1,..yn] of n values from N(M, S2)
there is a formula for 95% confidence interval = [y - k, y + k], where y is the observated mean
What is the probability that the mean is inside the confidence interval?
* frequentist: either 0 or 1
* baysian: 0.95

Example: p-value
===============
A coin has a Theta probability of landing heads. Want to test whether the coin is fair (versus the alternative that the coin lands heads with greated probability), that is wheteher Theta = 0.5

H0: Theta = 0.5
H1: Theta > 0.5

Will test at the alpha = 0.05 significance level

the test is H T T H H H H H H H H T

we compute p-value

if p <= 0.05: reject H0
else: we don't have enough evidence to reject H0

p = binomial_probability(tests)
p = 0.075
non reject H0

Actually I was flipping coins until I get three T, and I stopped at 12.
There more extreme case is that I will stop at 13 or more
now my p_value changes

p = (Y >= 12 | theta=0.5)
p = 0.0325
reject H0

What just happened?
==================
The data and model were identical we obtained different conclusions.
The difference came about in what was considered "as ormore extreme" than the observed data.
This can happen only in the Frequentist framework, but not in the Bayesian framework.

3. A coffee connoisseur claims to be able to tell the difference between whether cream is pured in coffee first, or vice versa.
In 10 trials, the person answers correctly 10 times. What do you think about the next trial?
From the frequentist point of view, the expected value is 100% he will be correct (it wouldn't really be 100% because a frequentist could use regularization)

NOTE: for machine learning for computational reasons you adapt a frequentist apporach, but the idea is that when you have a lot of data it doesn't matter whether you use bayesian or frequentist approach since your prior will be overwhelmed by data.

Bayes Theorem
=============

Suppose I have two events A and B, and they are temporally or causally related to each other.
The use of Bayes rule is to determine the probability of a past or cause event, given a current event has happened. Say something about the past based on what you know now.
P(A|B) is a `detection`,`forensic` probability, given what we now know (B) what is the probability that (A) would have happened?

Example: coins
==============
two coins, one fair one double-headed, placed in a box. One coin choosen and flipped. It lands heads, what is the probability that that coin is the double-headed coin?

A = double-headed selected
B = coin that selected lands heads
(A -> follows B)

P(A|B) ? probability of the past given what happened now
P(A) = 0.5
P(B|A) = 1
P(B|~A) = 0.5
P(A|B) = P(B|A) * P(A)/P(B) = 2/3

Example: Naive Bayes
====================
Task: assess the probability a new incoming e-mail is spam.
We identify the most frequent words common in spam.
W[j] is the even that j-th phrase appears in e-mail.
~W[j] does not appear in e-mail.

Suppose that an email contains 1st and 10th words, I want to compute
P(spam|W[0] ~W[1], .. , ~W[9], W[10])
P(W[0] ~W[1], .. , ~W[9], W[10] | spam) is hard to compute empirically, there might be few examples where exactly W[0] ~W[1], .. , ~W[9], W[10] , even worse if I assume n=100
Naive Bayes approach: assume independecy
P(W[0] ~W[1], .. , ~W[9], W[10] | spam)  =  P(W[0] | spam) * P(W[1] | spam) * P(W[2] | spam) * ... * P(W[n] | spam) 

[###################################################################################################################################################]
[##### how NB outperform other learners: homes.cs.washington.edu/~pedrod/papers/cacm12.pdf #########################################################]
[###################################################################################################################################################]

Bayesian Statistic
==================

> Unknown model parameters
parameters: uncertainty on parameters is described through probability distributions p(Theta), e.g. belief that a-priori the mean adult male weight is uniformly distributed from 160lbs to 200lsb.
> There is a probability prior-distribution over parameters, e.g. there is a really skewed gaussian on 0.5 my drunk friends tossing coin and I need huge amount of data to change my belief. I will have a posterior-distribution over my data.

> How data are generated based on the model parameters
data: observations are obtained from a ata-generating mechanism are assumed to follow a probability distribution p(y|Theta). For example, we may assume adult male weights are normally distributed around an unknown mean.

Can think of the model parameters coming first and data coming second


Steps of the Bayesian Analysis
==============================

1. Formulate a probability model for the data
2. Decide on a `prior distribution` for the unkown model parameters
3. Observe the data, and construct the likelihood function based on the data.
4. Determine the `posterior distribution` based on combination of prior and likelihood function. Is your state of knowledge based on the unknown parameters and this combination of prior and likelihood.
5. Summarize important features of the posterior distribution or calculate quantities of interest based on the posterior distribution.

Example: 30 day mortality of hearth attack
5 patients, 1 die, 4 surivive.

1. Formulate probability model
Bernoulli model.
Yi, survive = 0, dies = 1
P(Yi | y, Theta) = Theta for y = 1, 1 - Theta for y = 0
p(y|Theta) = pow(Theta, y) * pow(1 - Theta, 1 - y)

2. Choosing a prior
Uniform distribution for Theta, all values are equally plausible
p(Theta) = 1

3. Construct likelihood function
p(p1,p2,p3,p4,p5|Theta) = Theta * pow(1 - Theta, 4) = L(Tetha|y)
givining my relative believablity of my theta purely on my data alone
if you plot likelihood it is maximized on Theta = 0.2, but we are not maximizing the likelihood (that would be the frequentist approach)

4. Determine posterior distribution
Compromise between posterior distribution and likelihood (Bayes Rule)
P(Theta|Y) = p(Theta) * p(y | Theta) / Integral(p(Theta) * p(y|Theta) dTheta) ~ p(Theta) * p(y | Theta) ~ p(Theta) * Likelihood(Theta | y)
Operational definition of Bayes rule for statistical inference: Posterior ~ Prior * Likelihood

p(Theta|y) = c * 1 * pow(1 - Theta, 4)
turns out integration constant c is 30
p(Theta|y) = 30 * 1 * pow(1 - Theta, 4) -> Theta ~ Be(2,5) is a **Beta Distribution**

5. Summarize feature of the posterior distribution

Most believable value of theta is the mean of Be(2,5) = 2 / (2 + 5) = 2/7 = 0.286
Posterior mode
Central posterior interval for Theta
Highest posterior density (HPD)

*OBSERVATIONS:*
- Bayesian approach recognizes the asymmetry in inferences about Theta
- Same approach applies to multi-parameter models
- Frequency methods needs more advanced machinery but perform poorly over small dataset

Bayesian inference as a learning model -> this is because ofthe **sequential nature of Bayesian analysis**:
> Yesterday's posterior is today's prior

Example: 30 day mortality of hearth attack
5 patients, 1 die, 4 surivive.
What if they come one at a time? I can update my belief and update my posteriors -> prior patient by patient


Monte Carlo simulation
=====================
Why the whole world look like classical stastic?
When you get into complicated probabilistic models, when you get a posterior density is not easy to summarize.
If I get that is more complicated than a Beta, I would have problem in analysizing the distribution.
Instead of staring at posterior distribution, one way is to simulate values from that posterior distribution and then derives summaries from that distribution.

> Monte Carlo is use to summarize posterior distribution from a sample

Example: summarizingBeta(2, 5)
Simulate 10.000 values from Be(2, 5)
Report sample summaries from those 10.000 summaries

- *Predictive Inference*

You start from a posterior distribution, your goal is todetermine p(y~,y) that is the probability distribution for a new value y~ given the data we alread analyzed.

1. Simulated 10k values of Theta from the posterior distribution of Theta -> Thetas[10000]
2. for Theta in Thetas
       y~ = generate(Theta)
       predictions.append(y~)
3. summarize feature of predictions

Generative Models
=================

Statistical prediction: learn relatinoship between response and features through a machine learning algorithm
> generative model apporach to statistical prediction: propose a probabilistic mechanism in which data y are generated given x

Prior distribution + Data probability model = generative mechanism
(Usually generative model have latent variables,cascading generating values either of parameters or generating data)

Example of Generative Models:
- Hierarcical Linear Models
- Latent Dirichelet Allocation
- Generative Adversarial Networks
- Hidden Markov Models

Markov Chain Monte Carlo
========================
- Difficults to summarize posterios distribution
- If distributions are non normal is hard even to simulate with Monte Carlo

There is a very clever way to construct a Markov Chain:
given j unknown parameters, if I know all the other parameters and my data
If I can write a expression of every theta given all the other parameters, I can simulate only one parameter and then feed it inside the others.

p(Theta1|Theta2, Theta3.., y)
p(Theta2|Theta1, Theta3.., y)

This piping turns out is a Markov Chain, and if we keep iterating through we will be sampling for the complete posterior distribution. This process is called Gibbs sampler -> **JAGS** or **pyMC** software packages that implement MCMC Gibbs sampling.

All is needed is: specify generative model for the data, and hypearameters (burn-in) and they simulate.

Example: Bayesian Logistic Regression
Concentration of substance, number of beetles exposed, number of beetles killed/
1. Generative model for beetles death
y ~ Bin(n,p)
log(p) = alpha + beta(x) -> logistic regression
2. Choose a prior distributions
alpha = N(0, 10k) # on the intercept
beta = N(0, 10k) # on the multiplying concentration
3. Use bayesin Rule derive posterior distribution
4. Posterior distribution is too hard to summarize, I use JAGS sampling 
```JAGS CODE
linpred = alphastar + beta * ventered_x
for i in range(N):
	p[i] = ilogit(linpred[i]) # i logit is inverse logit
	yhat[i] = p[i] * n[i] # fitted values
    y[i] ~ dbin(p[i],n[i]) # model for y, binomial distribution, n[i] number of obs
```
This is my generative model and I call it from python
For a burn-in period I call my sampler (500 times)
Then I start MCMC sampling (3000 times) * 4 chaines
```Python CODE
beetles_model = pyjags.Model(beeltes_code,data)
beetles_burnin =  beetles_model.sample(500)
beetles_samples = beetles_model.sample(3000) 
```
5. I draw summaries by my 3000 * 4 simulates value
- posterior mean
- posterior std
- Rhat (convergence factor)

Hierarchical Linear Models
==========================

Cluster into different groupings, you know the labelling of the grouping in advance.
Example: number of days until pain goes away as a function of patient characteristics. The patients go to different cities, so it might be useful to take into account the different clustering of cities and build a different regression based on the cluster.

Example: study on reaction time and sleep deprivation.
Two extremes
- recognize that relationship between different patients: fit seperate least-square regression for each subject
Bad: very little data to evaluate
- I might be losing information in commonalities: pool all data togheter and fit a single least-square regression
Bad: does not recognize differences between the group

Main idea of Hierarchical Modelling is *having Seperate least-squares regression as a generative models.*

> **Hierarchical Modelling**: a compromise between spserate regressions and one overall regressino

Given G groups, I assume for every group my observations come from the same distribution.
All the coefficient are all coming from the same normal distribution and that's what connects everything together.
You take all this alphas and betas and you take a big robber bend and wrapping around them -> you are imposing a prior. They can not be not too far from each other.

> **Shrinkage Prior Distribution**: you force alpha and beta of linear regression not to stretch too far away from each other imposing a prior on them

Now alpha and beta come from the same family. You are gonna be informing from each individual subject perspective, I have a connection between all differenet groups.

Observation: shrinkage prior is actually a form or regularization. There is a connection between certain prior distribution (Laplace Distribution) and appplying a Lasso regularizer.


Probabilistic Topic Models
==========================

Unsupervised task: I want to cluster words in to documents as a way to represent different themes.

> LDA - Latent Dirichelet Allocation

- Documents exhibit multiple topics
- LDA is probabilistic modelwith corresponding generative process
- A topic is a distribution over a fixed vocabulary
- Only the number of topics is specified in advanced (like K-means)

















NOTES: khttps://en.wikipedia.org/wiki/Just_another_Gibbs_sampler
