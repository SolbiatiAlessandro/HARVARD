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
