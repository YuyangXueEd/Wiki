# Reinforcement Learning 
## Fundamentals of RL

### The K-armed Bandit Problem

#### The k-armed bandit

In the $k$-armed bandit problem, we have an agent who chooses between $k$ actions and receives a reward based on the action it chooses.

### Action-Values

The *value* is the **expected reward**:

$$
q_*(a)\overset{\cdot}{=}\mathbb{E}[R_t|A_t=a]=\sum_r p(r|a)r, \forall a\in \{1, \dots, K\}
$$

where $\overset{\cdot}{=}$ means "is defined as".

The goal is to maximise the expected reward:

$$
\arg\max_{a} q_*(a)
$$

We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$. We would like $Q_t(a)$ to be close to $q_*(a)$.

*Greedy*: If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We say that you are exploiting your current knowledge of the values of the actions.

If instead you select one of the nongreedy actions, then we say you are exploring, because this enables you to improve your estimate of the nongreedy action's value.

Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times.

### Action-value Methods

One way to estimate $q_*(a)$ is to average them. The **Sample-Average** method is to record the total reward for each action and divided by the number of times that action has been selected.

$$
Q_t(a)\overset{\cdot}{=}\frac{\sum^{t-1}_{i=1}R_i}{t-1}
$$

$$
A_t \overset{\cdot}{=}\arg\max_a Q_t(a)
$$

### Incremental update rule

$$Q_{n+1}=\frac1n \sum^n_{i=1}R_i=\frac1n(R_n + \sum^{n-1}_{i=1}R_i)=\frac1n(R_n+(n-1)\frac1{n-1}\sum^{n-1}_{i=1}R_i$$

$$
=\frac1n(R_n + (n-1)Q_n)=\frac1n(R_n+nQ_n-Q_n)=\frac1n (R_n + n Q_n - Q_n)
$$

$$
=Q_n+\frac1n (R_n - Q_n)
$$

$\mathrm{NewEstimate}\leftarrow \mathrm{OldEstimate + StepSize[Target-OldEstimate]}$

