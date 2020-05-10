---
layout:     post
title:      "Bellman Equation"
description: "How to solve Bellman equation using dynamic programming"
date:       2020-05-09 15:00:00
author:     "Yangyang Fu"
header-img: assets/img/posts/header-img/woman-typing-macbook.jpg

categories:
  - Learning/Reinforcement Learning
---

# Bellman Optimality Equations

Previous posts have drived the formulation for Bellman optimality equations. Bellman equations can be written in terms of state value and action value. The optimal policy can be found once we can find the optimal value functions, $v_{\*}$ or $q_{\*}$, which can be theoritically calculated as follows.

$$
\begin{equation}
\begin{aligned}
v_{*}(s) &=\max_{a}\mathbb E[R_{t+1}+\gamma v_{*}(S_{t+1})|S_t=s,A_t=a] \\
		&=\max_{a}\sum_{a}p(s',r|s,a)[r+\gamma v_{*}(s')] 
\end{aligned}
\end{equation}
\tag{1}\label{eq1}
$$

$$
\begin{equation}
\begin{aligned}
q_{*}(s,a) &= \mathbb E[R_{t+1}+\gamma \max_{a'}q_{*}(S_{t+1},a')|S_t=s,A_t=a] \\
		&=\sum_{s',r}p(s',r|s,a)[r+\gamma \max_{a'}q_{*}(s',a')] 
\end{aligned}
\end{equation}
\tag{2}\label{eq2}
$$

## Bellman Optimality Equation vs Optimal Policy

The ultimate goal of RL is to find the optimal policy. But how does the Bellman optimality equation relate to the optimal policy?

Once one has $v_{\*}$, it is relatively easy to determine an optimal policy. For each state $s$, there will be one or more actions at which the maximum is obtained in the Bellman optimality equation. Any policy that assigns nonzero probability only to these actions is an optimal policy. You can think of this as a one-step search. If you have the optimal value function, $v_{\*}$, then the actions that appear best after a one-step search will be optimal
actions. 

Another way of saying this is that any policy that is greedy with respect to the
optimal evaluation function $v_{\*}$ is an optimal policy. The term greedy is used in computer
science to describe any search or decision procedure that selects alternatives based only
on local or immediate considerations, without considering the possibility that such a
selection may prevent future access to even better alternatives. Consequently, it describes
policies that select actions based only on their short-term consequences. The beauty of $v_{\*}$ is that if one uses it to evaluate the short-term consequences of actions—specifically, the one-step consequences—then a greedy policy is actually optimal in the long-term sense in which we are interested because $v_{\*}$ already takes into account the reward consequences of all possible future behavior. By means of $v_{\*}$, the optimal expected long-term return is turned into a quantity that is locally and immediately available for each state. Hence, a
one-step-ahead search yields the long-term optimal actions. 

Having $q_{\*}$ makes choosing optimal actions even easier. With $q_{\*}$, the agent does not
even have to do a one-step-ahead search: for any state $s$, it can simply find any action
that maximizes $q_{\*}(s,a)$. The action-value function ectively caches the results of all
one-step-ahead searches. It provides the optimal expected long-term return as a value
that is locally and immediately available for each state–action pair. Hence at the cost of representing a function of state–action pairs, instead of just of states, the optimal actionvalue
function allows optimal actions to be selected without having to know anything
about possible successor states and their values, that is, without having to know anything
about the environment’s dynamics.

Next, how can we solve the Bellman optimality equation theoretically and numerically? 

## Theoretical Solutions

Explicitly solving the Bellman optimality equation is one way to find the optimal policy. To guanratee the Bellman optimality equation can be directly solved, at least three assumptions need to be awared of:

1. The dynamics of the environment $p(s',r|s,a)$ are accurately known.
2. There always are sufficent computational resources.
3. The environment has Markov property.

Now let's assume our system can meet all the above assumptions. 

As we can see from the Bellman optimality equation, to find $v(s)$, we need to know all the values of the succedding states $s'$ after $s$, $v(s')$. Therefore, the intertwinement give us a set of equations that should be solved simultenaously. The number of equations in the set equals to the number of states, $\vert S$.

More details are given in the example below to see how the set of equations can be formed.

# to-do
1. How to solve it theotically
2. An example
3. Conclusions
4. Next Step
