second part of the semester is going to be tough! hope you all enjoyed your break :))

INTRODUCTION TO REINFORCEMENT LEARNING	(RL)
============================================
[Lapan, Maxim. Deep Reinforcement Learning Hands-On]

**What is RL**: Mouse in a maze, mouse (agent) can move in the maze (environment), in the maze there are blocks with cheese (rewards) and blocks with electroshock, the mouse wants the cheese but not the electroshock, mouse can observe environment (observation).
> RL is make *sequential decisions* in an environment so as to maximize some notion of overall rewards acquired along the way.

There are multiple scenarios that the mouse can undertake, e.g stay still or take a lot of electro-shocks to get a lot of cheese. Every scenario yields a different outcome.
> Simple Machine Learning problems have a *hidden time dimension*, which is often overlooked, but is actually important in a production system
> *RL incorporates time into learning*, which puts it much close to the human perception of artificial intelligence.

What we don't want the mouse to do? We do not want to have best actions to take in every specific situation, not flexible enough.
Similar problem as us planning for our day. We need some 'magic' set of methods that will allow our mouse to learn on its own how to avoid electricity and gather as much as possible. RL is that 'magic' tool box.

**Challenges of RL**:
If action decides to do stupid things,then the observations will tell nothing about how to improve the outcome (only negative feedback)
> Observations depends on agent's actions. 
We only get a reward, but you don't have a *gradient* to explore the space, like when you teach your children and you guide them through the process and you guide them through the process.
> Agents need to not only explot the policy they have leaerned,but to actively explore the environment. This is known as *exploration vs exploitation.*
Maybe by doing things differently we can significantly improve the outcome. E.g If I am looking for a restaurant I can use tripadvisor or just explore the city. This interplay is the same as in a RL environment, and is not set: you need to play around with it.
> (missing)

**RL formalisms and relations**
* *Agent*: somebody or something that interact with environment by executing actions, taking observations and recieving rewards
* *Environment*: the universe outside the agent
* *Actions*: things that agent can do in the environment, you can not invent your own actions they are set.
* *Rewards*: scalar signed value, purpose is to tell our agent how well they behave, reward or *reinforced* the behaviour, grades are a reward system to give you feedback about how well you do in class
* *Observations*: everything else you get from the environment beside reward is observation, we have them for convenience

> All goals can be described by the maximization of some expected cumnulative rewards

**Difference between RL and other ML paradigms**
* No supervision: is not unsupervised learning nor supervised learning, not totally unguided but is in between.
* Feedback is sometimes delayed
* Time matters
* Feedback

Markov Decision Processes (MDP)
===============================

Example:
* *System*: weather in Boston, for a given system we observe states
* *States*: sunny/rainy
* *History*: sequence of observations [sunny, sunny, rainy, sunny, ..] called Markov Chain

You create a discrete and finite number of states, this might be a problem since might be hard to discretize.
The system fullfills the Markov Property.

> The future system dynamics from any state have to depend on this state only

In the weather example this means that the proability of tomorrow being sunny is only dependent on today weather and not about the past.

> Transition probabilities are expressed as a *transition matrix*, which is a square matrix of size NxN where N is the number of statesin our model.

We want to extend Markov Process to include rewards. We add a second square matrix which matrix(j)(k) expresses rewards of going from state j to state k

> For every time point, we define *return* as a sum of subsequent rewards def G(t): sum([return(time) for time in [t+1, ..., t+N]])

Actually we want to add a discount factor to take into account how far we are from the current time step. 
def G(t): sum([ pow(discount_factor, step) * return(time) for step, time in enumerate([t+1,...,t+N])])

Since there are probabilities to reach other states this can vary a lot depending which path we take, we can take the expecation of return for any state.

> Value of state is definied as V(s) = ExpectedValue(G | state == s)

Value of State is our expected return given the current state.
How can we extend MRP to include actions? We can adda finite set of actions A, called agent's action space.
We add an extra dimension to our transition state, now is a three dimensional matrix NxNxM where M = len(A)
Why will the action change the transition probability? Because the rewards depends on your action. Also actions are not deterministic. There are different actions that could lead you to the same state.

> Rewards depends both on the state and also on the action that leads to that state

If you pass the exam (state A) you can either pass it by studying (action A) or by not studying (action B). 
reward(state A, action A) != reward(state A, action B)

> Policy is some sets of rules that control the behaviour of the agent

Even for a simple environment we can have a variety of policies (always go forward, always go backward, ..)
Different policies gives different returns. *The goal is to find the policy that optimize the return*

> Formally, policy is the probability distribution over actions for every possible state.

Policies can be learnt.

> Bellman Equation (Deterministic) for computing policies return. Value(s) = max([return(state) + attenutation * Value(state) for state in neighbour(s)])

The idea of the above is to maximize the next step, in the transiation from a state to another.
We can generalize this adding probabilites, it becomes Bellman Equation (Stochastic).

(check out course Harvard CS282 for more RL)
RL: set of actions that you can take, policies tell you what acitons to take. Agent, feedback, env. We want to find the best policy to get the maximum reward.

Value is the forward looking total reward.
Lets start with state S0. Value at S0 given that I take action Ai: 
V0(A = Ai) = Ri + gammaVi
           = (immediate reward) + (compounded future reward)

maximum value
V0 = max([Ra + gammaVa] for a in range(1, N))

this if it is deterministic, in real case the result of an action is not deterministic but probabilistic

V0 = max([ sum([ p(action, S0 -> s)(Rs + gammaV) for s in state]) for a in actions])

Value of Action Q(s, a)
=======================

the total reward of the one-step rewards for taking action A in state S can be defined via Value(S)
provides a convenient form for policy-optimization and learning policies Q-learning

Q(state, action) = R(state, action) + gamma * Expected(Value(state + 1))

Value function are recursive -> Dynamic Programming!

Model Based method: Knowing the transition matrix
=================================================
* Value iteration

1. start with some aribtrary value assignments V(0)[S]
2. Update policy andrepeat until abs(V(n+1)[s] - V(n)[s]) < 1e-6

update:
Q(n)[state, action] = Reward[state, action] + gamma * ExpectedValue(Value[state`])
Value(n + 1)[state] = max([Q[n](state, action) for action in actions])
Policy(n)[state] = argmax([Q[n](state, action) for action in actions]) # the state, action that gives me maximum value is my policy

INTUITION: iteratively improve your value estimates using Q, V relations.

* Policy iteration

1. initialize policy
2. update policy based on Vs

Model Free: Not knowing the transition matrix


Model Free method: not knowing the transition matrix
====================================================
Model-free settings are closer to real life settings
new factor

* ON-Policy Learning (SARSA)

Transition: (state, action) -> (next_state, reward)
you keep updating this table with the data you get, table is called Q-Table

Q(s,a) = Q(s,a) + alpha[ R(s,a) + gamma * Q(s',a' ~ pi) - Q(s, a) ]

alpha: blending factor, I am blending the new data. Eventually Q(s', a' ~ pi) and Q(s, a) will converge, since stochasticity is get embedded as the process goes on
                      
is called on-policy because it use the current estimate of the optimal policty to genereate the behaviour
exploration-exploitation problem is faced using a *epsilon-greedy policy*, you don't want to go fully greedy

* Q-Learning

You go fully greedy, epsilon-greedy policy with epsilon is zero.

Q(s,a) = Q(s,a) + alpha[ R(s,a) + gamma * max( [ Q(s',action ~ pi) for action in actions]) - Q(s, a) ]

q-learning is updated in a more greedy way, the difference is it is looking over the actions to get the more optimistic updated. I take maximum of everything is possible. Regularizing the reinforcement learning agency.
We alpha we want to slowly blend from our initial assignment, we don't want to dump.


Advanced Section
================
[Srivatsan Srinivasan]

Q-Learning:
- off-policy, model-free
- tabluar q-learning not scalable
- approximate q-function

Parametric Q-Learning
====================
Use a function approximator to estimate the action-value function
Q*(s,a) ~ Q(s,a,Theta)
TASK: we want to find a functino approximation for Q satisfying Bellman Equation

define a target: Q+(s,a,Theta|s,a,s') ~ R(s,a) + gamma*max(Q(s,action,Theta) for action in actions')
target is what you want your Q value to be if the original Bellman equation was true
loss penalize how much you are going away from Bellman equation
batch is a set of transitions

Use gradient descend to learn target function

Value based Deep RL (DQN)
========================
Games: Each frame is a state, transitions is from frame to frame
Q(s,a,Theta) neural network with weights Theta

Potential Issues:
- Learning from batches of consecutive samples is problematic
- current Q net params determines next training samples -> can lead to bad feedback loops
- Samples are correlated -> inefficient learning
If you updated weights of the nature very frequently target become non stationary

HACK1:
experience replay buffer
train Q-network on random minibatches from replay buffer and not from live updates

HACK2:
Target network: another NN (Theta-) that is more stationary than the current Q-Network (Theta)
Use (Theta-) as a lagged version of Theta which is updated periodically after specified numberof iterations 


Double DQN (DDQN - 1)
=====================
problem: over estimate of q values because same net is doing both

decompose the max operation in TD updated into action selectino and action evaluation
QNet select action
Target Network evalutes the value
