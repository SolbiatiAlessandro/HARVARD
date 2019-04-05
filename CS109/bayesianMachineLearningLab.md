Overview of Bayesian Analysis
=============================

> 1. Formulate a **model**
> 2. Define a **prior** distributions of unknown parameters
> 3. Construct **likelihood** function based on observed data
> 4. Determine the **posterior** distribution
> 5. **Summarize** from posterior distribution

Example
1. Model: all coin flips return heads with a probability theta
2. Prior: theta has a uniformly distributed probability between 0 and 1, init at 0.5
3. Likelihood: Construct a likelihood based on observed data (HTHTHTHT -> theta * ( 1 - theta) * .. )
4. Posterior: posterior is proportional to prio x likelihood or Posterior = c * prior * likelihood

likelihood here means the likelihood that the sampled theta is the real theta from which data were drawn from..
Given this likelihood (a number) I updated my belief with Bayes rule given my prior.

5. Fin mean value for theta

How to calculate the posterior distribution?
============================================

In most cases, is not possible to analytically summarize your posterior distribution.
That's where **Monte Carlo simulations** come in. Take a very large samplefrom the posterior distribution and use sample summaries as approximate actual summaries.

What about **Markov Chain Monte Carlo MCMC**?
Posterior densities are too complex/non-standard that even MC simulation becomes hard.
Markov chain has a stationarydistribution which is the same as the target distribution. running a morkov chain long enough will converge it to the target distribution - in this case the posterior distribution.

How to run a MCMC?
[Gibbs Sampler](https://en.wikipedia.org/wiki/Gibbs_sampling), [JAGS](https://en.wikipedia.org/wiki/Just_another_Gibbs_sampler) and [PyJags](https://pyjags.readthedocs.io/en/latest/) or PyMC3


Lab notebook walkthorugh
========================
([notebook here](https://github.com/Harvard-IACS/2019-CS109B/blob/master/content/labs/lab8/cs109b_lab8.ipynb))

Let's use JAGS to estimate how fair a coin is, based on 100 coin flips

1. We generate some artificial data with a `true_theta` of  0.6
2. We use `pyjags.Model` for our coinflip process, using a `init_theta` (prior) of 0.5. In `coinflip_code` we express our prior belief (in R) that the theta distribution is a uniform, where we use R function `dbern` to express that a single coin flip is a trial of a Bernoulli with a theta
3. We simulate our process first with a burn-in period and then with a sampling period
[Burn-in is intended to give the Markov Chain time to reach its equilibrium distribution, particularly if it has started from a lousy starting point. To "burn in" a chain, you just discard the first 𝑛 samples before you start collecting points.]
4. After we simulate we see a traceplot were on the X axis there are iteration, and on Y axis there is the current `theta` estimate
5. The mean of this `theta` estimations will be my inferred value for `theta`

Lab2 notebook walkthourgh
=========================
([notebook here](https://github.com/Harvard-IACS/2019-CS109B/blob/master/content/labs/lab9/cs109b_lab9_student.ipynb))

whole course on this topics [AM207 Advanced Scientific Computing: Stochastic Optimization Methods. Monte Carlo Methods for Inference and Data Analysis](http://iacs-courses.seas.harvard.edu/courses/am207/)

**Schools Data and Bayesian Modeling**
We need to answer the question: do we want to implement this program or not?
A hierarchical model for the schools:
This program intrisically has a spread of effects when school implmements. Different implementations will have a different range of effect.
This range of effect is defined by MU and TAU, where
MU ~ Uniform(-20,20) -> MU is the mean of the distribution of the observed value
TAU ~ Uniform(0, 10) -> TAU is the variance of the distribution of the observed value
Sigma-j given

> **y-j ~ Normal(Mean=theta-j,SD=sigma-j)** : First, the observed data (one value per school) are 1) normally distributed 2) each centered at a different value, one per school.
> **theta-j ~ Normal(Mean=mu, SD=tau)** : Parrameters are the 'true average effect' of the program in school j, separate from the (noisy) effect we actually observe.

I code this into Jags
```
schools_model_code = '''
model {
    
    mu ~ dunif(-20,20)
    tau ~ dunif(0,10)
    
    for (j in 1:J){
        theta[j] ~ dnorm(mu, 1/pow(tau,2))
    }
    
    for (j in 1:J){
        y[j] ~ dnorm(theta[j], 1/pow(sigma[j],2))
    }
}
'''
```
I generate my samples and then I draw summaries (read lectures for more info on this about MCMC baysianMachineLearning.md)

If I draw a summary on School A on the posterior distribution, the mean of the distribution on school A is around ~10 -> the starting observation on school A was ~28! This is the shirknage discussed in lecture

What does doing a Bayesian anlysis buy? What is the big difference with frequentist?

- Bayesian analysis give you a probability distribution over your parameters, all of the possible values of your Theta and Tau

What the cost of doing a Bayesian Analysis?

- PRIOR, you need to be  willing to introduce a prior: if we start from different prior we get different results.



