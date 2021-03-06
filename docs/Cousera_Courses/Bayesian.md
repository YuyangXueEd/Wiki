# Bayesian Statistics: From Concepts to Data Analysis

## Lesson 1: Probability

### Backgrounds

#### Rules of Probability

If $A$ and $B$ are two events, the probability that $A$ or $B$ happens is the probability of the union of the events:

$$
P(A\cup B)=P(A) + P(B) - P(A\cap B)
$$

If a set of events $A_i$ for $i = 1, \dots, m$ are mutually exclusive (only one can happen), then:

$$
P(\bigcup^m_{i=1}A_i) = \sum^m_{i=1}P(A_i)
$$

#### Odds

Suppose we denote rolling a "$4$" on a fair six-sided die as the event $A$. Then $P(A)=\frac16$. The odds for events $A$, denoted $O(A)$ is defined as $O(A) = P(A)/P(A^c)=P(A)/(1-P(A))$:

$$
O(A)=\frac{1/6}{5/6}=\frac15
$$

This can also be expressed as $1:5$ (or $5:1$ "odds against").

If event $B$ has $a:b$ odds, then $P(B)/(1-P(B))=a/b \Rightarrow P(B)=a/(a+b)$.

#### Expectation

The expected value of a random variable $X$ is a weighted average of values $X$ can take, with weights given by the probabilities of those values. If $X$ can take on only a finite number of values, we can calculate the expected value as:

$$
E(X) = \sum^n_{i=1} x_i \cdot P(X=x_i)
$$

If $X$ is a continuous random variable with probability density function $f(x)$, we replace the summation with an integral:

$$
E(X)=\int^\infty_\infty x\cdot f(x)dx
$$

Let $X$ and $Y$ be random variables with $E(X)=\mu_X$ and $E(Y)=\mu_Y$. Suppose we are interested in a new random variable $Z=aX+bY+c$. The mean of $Z$ is easy to compute:

$$
E(Z)=E(aX+bY+c)=aE(X)+bE(Y)+c = a\mu_X+b\mu_Y+c
$$

We can also compute expectations of functions of $X$. We can have:

$$
E(g(x))=\int^\infty_\infty g(x)f(x) dx
$$

#### Variance

The variance of random variable measures how spread out its values are. If $X$ is a random variable with mean $E(X)=\mu$, then the variance is $E[(X-\mu)^2]$. The variance is the expected value of the squared deviation of $X$ from its mean.

$$
\mathrm{Var}(X)=\int^\infty_\infty(x-\mu)^2dx
$$

A convenient formula for the variance is:
$$
\mathrm{Var}(X)=E[X^2]-(E[X])^2
$$

Let $X$ and $Y$ be random variables with $\mathrm{Var}(X)=\sigma^2_X$ and $\mathrm{Var}(Y)=\sigma^2_Y$. Suppose we are interested in a new random variable $Z=aX+bY+c$. The mean of $Z$ is easy to compute:

$$
\mathrm{Var}(Z)=\mathrm{Var}(aX+bY+c)=a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y)+0 = a^2\sigma^2_X+b\sigma^2_Y
$$



### Classical and Frequentist Probability

- Classical: equally likely
- Frequentists: relative frequency

### Bayesian Perspective

- Bayesian: Personal Perspective

## Lesson 2: Bayesian Theorem

### Conditional Probability

$P(A|B)=\frac{P(A\cap B)}{P(B)}$

|        | Female | Not Female | Total |
|:------:|:------:|:---------:|:-----:|
|   CS   |    4   |     a     |   12  |
| Not CS |    b   |     c     |   d   |
|  Total |    9   |     e     |   30  |

- 30 students, 9 females, 12 computer science of which 4 females.
	- Male computer science student: 8
	- Not CS students: 18
	- Not CS females: 5
	- Not CS male students: 13
	- Total male students: 21

|        | Female | Not Female | Total |
|:------:|:------:|:----------:|:-----:|
|   CS   |   4    |     8      |  12   |
| Not CS |   5    |     13     |  18   |
| Total  |   9    |     21     |  30   | 

$$
P(F)=\frac{9}{30}=\frac{3}{10},\ P(CS) = \frac{12}{30}=\frac{2}{5},\ P(F\cap CS) = \frac{4}{30} = \frac{2}{15}
$$

Then, as for conditional problem:

$$
P(F|CS) = \frac{P(F\cap CS)}{P(CS)} = \frac{4/30}{12/30} = \frac13,\ P(F|CS^c) = \frac{P(F\cap CS^c)}{P(CS^c)} = \frac{5/30}{18/30}= \frac{5}{18}
$$

#### Independence

$$
P(A|B) = P(A) \Rightarrow P(A\cap B) = P(A)P(B)
$$

### Bayes' Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|A^c)P(A^c)} = \frac{P(A \cap B)}{P(B)}
$$


$$
P(A_1|B) = \frac{P(B|A_1) P(A_1)}{\sum^m_{i=1}P(B|A_i)P(A_i)},\ \sum^m_{i=1}P(A_i)=1
$$

- Example: 
	- $P(+|HIV) = 0.977$
	- $P(-|NoHIV) = 0.926$
	- $P(HIV) = 0.0026$
	- What is $P(HIV|+)$ ?

$$
P(HIV|+) = \frac{P(+|HIV)P(HIV)}{P(+|HIV)P(HIV) + P(+|NoHIV)P(NoHIV)} = \frac{0.977 \times 0.0026}{0.977 \times 0.0026 + (1-0.926) \times (1-0.0026)}=0.033
$$

### Bayes Theorem for continuous distributions

When dealing with a continuous random variable $\theta$, we can write the conditional density for $\theta$ given $y$ as:

$$
f(\theta|y)=\frac{f(y|\theta)f(\theta)}{\int f(y|\theta)f(\theta)d\theta}
$$

## Lesson 3: Review of Distributions

### Discrete Distributions

#### Geometric

The geometric distribution is the **number of trails needed to get the first success**, i.e., the number of Bernoulli events until a success is observed.

$$
X\sim \mathrm{Geo}(p), E[X]=\frac1p
$$

$$
P(X=x|p)=p(1-p)^{(x-1)},\ \mathrm{for\ }x=1,2, \dots
$$

#### Bernoulli

$$
X \sim B(p),\ P(X=1)=p,\ P(X=0)=1-p
$$

$$
f(X=x|p) = f(x|p) = p^x (1-p)^{1-x}I_{\{x\in \{0, 1\}\}^{(x)}}
$$

Expected value:

$$
E[X] = \sum_x xP(X=x)=(1)p + (0)(1-p)=p
$$
Variance: 
$$
\mathrm{Var}(x) = p(1-p)
$$

#### Binomial

The distribution of two possible outcomes.

$$
X \sim B_{in}(n, p)
$$

$$
P(X=x|p)=P(x|p)=(\frac{n!}{x!(n-x)!})p^x(1-p)^{n-x}, \mathrm{for}\ x\in \{0, 1, \dots, n\} 
$$

Expected Value:

$$
E[X] = np
$$

Variance:

$$
\mathrm{Var}(X) = np(1-p)
$$

#### Multinomial

Another generalisation of the Bernoulli and the binomial is the multinomial distribution, which is like a binomial when there are more than two possible outcomes. 

Suppose we have $n$ trials and there are $k$ different possible outcomes which occur with probabilities $p_1, \dots, p_k$.

$$
f(x_1, \dots, x_k|p_1, \dots, p_k)=\frac{n!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}
$$

#### Poisson

The Poisson distribution is used for counts, the parameter $\Gamma>0$ is the rate at which we expect to observe the thing we are counting:

$$
X\sim Pois(\lambda)\ \ E[X]=\lambda,\ \ \mathrm{Var}[X]=\lambda
$$

$$
P(X=x|\lambda)=\frac{\lambda^x\exp(-\lambda)}{x!} \ \ \mathrm{for}\  x=0, 1, 2,\dots
$$

A Poisson process is a process wherein events occur on average at rate $\lambda$, events occur one at a time, and events occur independently for each other.

### Continuous Distributions

#### Uniform 

The PDF is sort of proportional to the probability that the random variable will take a particular value. 

The key idea is that if you integrate the PDF over an interval, it gives you the probability that the random variable would be in that interval.

$$
X \sim U[0, 1]
$$

$$
f(x) = \begin{cases}
1 & \mathrm{if,} \ x\ \in [ 0,\ 1]\\
0 & \mathrm{otherwise}
\end{cases} = I_{\{0 \leq x \leq 1\}^{(x)}}
$$

![bayesian_uniform.png](../_media/bayesian_uniform.png)

The probability between $0<x<\frac12$:

$$
P(0<x<\frac12)=\int^{\frac12}_0 f(x)dx = \int^{\frac12}_0 dx = 0.5
$$

$$
P(0\leq x\leq\frac12)=\int^{\frac12}_0 f(x)dx = \int^{\frac12}_0 dx = 0.5
$$

Because there are an infinite number of possible outcomes, so for the possibility of a certain value is 0.

$$
P(X=\frac12)=0
$$

Expectation value:

$$
E[x]=\int^\infty_{-\infty}xf(x)dx
$$

if $X \bot Y$:

$$
E[X + Y] = E[X] + E[Y], E[XY]=E[X]E[Y]
$$

A more general case:

$$
X \sim Uniform(a+b),\ E[X]=\frac{a+b}{2},\ \mathrm{Var}[X]=\frac{(b-a)^2}{12}
$$

$$
f(x|\theta_1, \theta_2)= \frac{1}{\theta_2-\theta_1}I_{\theta_1 \leq x \leq \theta_2}
$$


#### Exponential Distribution

The exponential distribution is often used to model the waiting time between random events. We can write $X$ follows an exponential distribution with a rate parameter $\lambda$.

$$
X \sim \mathrm{Exp}(\lambda),\ f(x|\lambda)=\lambda e^{-\lambda x}
$$

$$
E[X]=\frac{1}{\lambda}, \mathrm{Var}(x)=\frac{1}{\lambda^2}
$$

#### Gamma

If $X_1, X_2, \dots, X_n$ are independent (and identically distributed $\mathrm{Exp}(\lambda)$) waiting times between successive events, then the total waiting time for all $n$ events to occur $Y=\sum^n_{i=1}X_i$ will follow a gamma distribution with shape parameter $\alpha=n$ and rate parameter $\beta=\lambda$.

$$
Y \sim \mathrm{Gamma}(\alpha, \beta), \ E[Y]=\frac\alpha\beta, \ \mathrm{Var}[Y]=\frac\alpha{\beta^2}
$$

$$
f(y|\alpha, \beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y}I_{y\geq 0}(y)
$$

where $\Gamma(\cdot)$ is the gamma function, a generalisation of the factorial function which can accept non-integer arguments. If $n$ is a positive integer, then $\Gamma(n)=(n-1)!$

It is used to model positive-valued, continuous quantities whose distribution is right-skewed. As $\alpha$ increases, the gamma distribution more closely resembles the normal distribution.

#### Beta

The beta distribution is used for random variable which take on values between $0$ and $1$.

$$
X\sim Beta(\alpha, \beta),\ E[X]=\frac{\alpha}{\alpha+\beta},\ \mathrm{Var}[X]=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

$$
f(x|\alpha,  \beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma{\alpha}\Gamma{\beta}}x^{\alpha-1}(1-x)^{\beta-1}I_{\{0<x<1\}}(x)
$$

where $\Gamma(\cdot)$ is the gamma function introduced with the gamma distribution.

#### Normal Distribution

$$
X \sim N(\mu, \sigma^2)
$$

$$
f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\{-\frac{1}{2\sigma^2}(x-\mu)^2\}
$$

$$
E[X]=\mu, Var(x)=\sigma^2
$$

If $X$ and $Y$ are independent random variables:

$$
X\sim N(\mu_x, \sigma_x^2),\ Y\sim N(\mu_y, \sigma_y^2),\ Z=X+Y,\ Z\sim N(\mu_x + \mu_y, \sigma_x^2+\sigma_y^2)
$$

If not, we still have 

$$
E(X+Y)=E(X) + E(Y)
$$

but now

$$
\mathrm{Var}(X+Y)=\mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y)
$$

where

$$
\mathrm{Cov}(X, Y)=E[(X-E[Y])(Y-E[X])]
$$

If $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$ are independent, then $X_1+X_2 \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$. Consequently, if we take the average of $n$ independent and identically distributed normal random variables $\bar{X}=\frac1n \sum^n_{i=1}X_i, X_i \sim N(\mu, sigma^2)$, then $\bar{X}\sim N(\mu, \frac{\sigma^2}{n})$.

#### t

We can get from normal distribution:

$$
\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\sim N(0,1)
$$

If we estimate the unknown $\sigma$ from data, we can replace it with $S=\sqrt{\sum_i(X_i-X)^2/(n-1)}$, the sample standard deviation. This causes the expression above to no longer be distributed as standard normal, but as a standard $t$ distribution with $v=n-1$ degrees of freedom:

$$
Y \sim t, \ E[Y]=0 \mathrm{if\ }v>1,\ \mathrm{Var}[Y]=\frac{v}{v-2} \mathrm{if\ }v>2 
$$

$$
f(y)=\frac{\Gamma(\frac{v+1}{2})}{\Gamma(\frac v2)\sqrt{v\pi}}(1+\frac{y^2}{v})^{-(\frac{v+1}{2})}
$$

The t distribution is symmetric and resembles the normal distribution, but with thicker tails. As the degrees of freedom increase, the t distribution looks more and more like the standard normal distribution.

### Central Limit Theorem

With sufficiently large sample sizes, the sample average approximately follows a normal distribution. This underscores the importance of the normal distribution, as well as most of the methods commonly used which make assumptions about the data being normally distributed.

## Lesson 4: Frequentist inference

### Confidence intervals

Example: A coin was flipped 100 times: 44 heads, and 56 tails.

$x_i \sim B(p)$ by Central Limit Theorem:

$$
\sum_{i=1}^{100}x_i \overset{\cdot }{\underset{\cdot }{\sim }} N(100p, 100p(1-p))
$$

By the properties of normal $95\%$ of the time. We'll get a result within 1.96 standard deviations of the mean. Thus we can say $95\%$ of the time, we expect to observe between $100p-1.96\sqrt{100p(1-p)}$ and $100p + 1.96\sqrt{100p(1-p)}$.

We observe that $\hat p=\frac{44}{100}=0.44$, so that the confidence interval (CI) is: $44 \pm 9.7$, which means $(34.3, 53.7)$.

we can say that we're $95\%$ confident that p is in the interval $(0.343, 0.537)$.

### Likelihood function and maximum likelihood

$Y_i \overset{iid}{\sim} B(\theta),\ P(Y_i=1)=\theta$.

$$
P(\underset{\sim}{Y}=\underset{\sim}{y}|\theta) = P(Y_1=y_1)\cdots P(Y_n=y_n)=\prod^n_{i=1}P(Y_i=y_i|\theta)=\prod^n_{i=1}\theta^{y_i}(1-\theta)^{1-y_i}
$$

The likelihood:

$$
L(\theta|\underset{\sim}{y})=\prod^n_{i=1}\theta^{y_i}(1-\theta)^{1-y_i}
$$

One way to estimate $\theta$ is that we choose the $\theta$ that gives us the largest value of the likelihood. It makes the data the most likely to occur for the particular data we observed. This is referred to as the maximum likelihood estimate or **MLE**.

$$
\hat{\theta} = \arg\max L(\theta|\underset{\sim}{y})
$$

Or Maximum log likelihood:

$$
\ell(\theta) = \log L(\theta|\underset{\sim}{y}) = \sum \left[y_i\log\theta + (1-y_i)\log(1-\theta)\right] = \left(\sum y_i\right)\log \theta + \left(\sum(1-y_i)\right)\log(1-\theta)
$$

Computing the MLE:

$$
\ell'(\theta) = \frac1{\theta}\sum y_i \frac{1}{1-\theta} \sum (1-y_i) = 0
$$

$$
\hat{\theta} = \frac1n \sum y_i = \hat{p}
$$

#### Derivative of a Likelihood: Normal distribution

Consider the normal likelihood where only the mean $\mu$ is unknown:

$$
f(y|\mu) = \prod^n_{i=1}\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac1{2\sigma^2}(y_i-\mu)^2\right] = \frac{1}{(\sqrt{2\pi\sigma^2})^n}\exp\left[-\frac1{2\sigma^2}\sum^n_{i=1}(y_i-\mu)^2\right]
$$

which yields the log-likelihood:

$$
\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left[\sum^n_{i=1}y_i^2 - 2\mu\sum^n_{i=1}y_i+n\mu^2\right]
$$

We take the derivative of $\ell(\mu)$ with respect to $\mu$ to obtain:

$$
\frac{d\ell(\mu)}{d\mu} = \frac{\sum^n_{i=1}y_i}{\sigma^2} - \frac{n\mu}{\sigma^2}
$$

We obtain 