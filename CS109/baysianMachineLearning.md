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
P(A|B) = P(B|A)*P(A)/P(B) = 2/3

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
4. Determine the `posterior distribution`.
5. Summarize important features of the posterior distribution or calculate quantities of interest based on the posterior distribution.

NOTES: khttps://en.wikipedia.org/wiki/Just_another_Gibbs_sampler
