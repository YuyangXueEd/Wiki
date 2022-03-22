## Introduction

Laplacian approximations (LAs) [^1] are popular UQ methods that are used to estimate Bayesian inference. 

## Method

They build a Gaussian distribution around the true posterior using a Taylor expansion around the MAP, $\theta^*$, as follows:

$$
p(\theta|D)\approx p(\theta^*)\exp \{-\frac12 (\theta-\theta^*)'\mathbb{H}_{|\theta^*}(\theta-\theta^*)\}
$$

where $\mathbb{H}_{|\theta^*} = \triangledown_\theta p(y|\theta)\triangledown_\theta p(y|\theta)'$ indicates the Hessian of the likelihood estimated at the MAP estimate.

- Ritter et al. [^2] introduced a scalable LA (SLA) approach for different NNs.
	- They proposed the model, then compared it with other well-known methods such as dropout and a diagonal LA for uncertainty estimation in the networks.
- Feng et al. [^3], with the help of conditional random fields (CRFs) on top of BNNs, could determine contextual information and carry out semisupervised learning
	- the author compared the performance of LA with a variant of MC dropout.
- Lee et al. [^4] used an LA-based inference engine for natural parameters and information in the form of a Gaussian distribution.
	- They managed to scale LA on the ImageNet dataset by spending considerable time tuning the hyperparameters so that they could make a meaningful comparison.
- Humt et al. [^5] applied existing BO techniques to tune the hyperparameters of LA.
	- The outcomes indicated that the proposed Bo approach required fewer iterations than random search.

## Applications

## Reference

[^1]: D.J. MacKay, D.J. Mac Kay, [Information Theory, Inference and Learning Algorithms](https://www.cambridge.org/gb/academic/subjects/computer-science/pattern-recognition-and-machine-learning/information-theory-inference-and-learning-algorithms?format=HB&isbn=9780521642989), Cambridge University Press, 2003.

[^2]: H. Ritter, A. Botev, D. Barber, [A scalable laplace approximation for neural networks](https://discovery.ucl.ac.uk/id/eprint/10080902/), in: 6th international conference on learning representations, in: ICLR 2018-Conference Track Proceedings, Vol. 6, International Conference on Representation Learning, 2018.

[^3]: J. Feng, M. Durner, Z.-C. Marton, F. Balint-Benczedi, R. Triebel, [Introspective robot perception using smoothed predictions from Bayesian neural networks](https://link.springer.com/chapter/10.1007/978-3-030-95459-8_40), 2019.

[^4]: J. Lee, M. Humt, J. Feng, R. Triebel, [Estimating model uncertainty of neural networks in sparse information form](https://arxiv.org/abs/2006.11631), 2020, arXiv preprint arXiv:2006.11631.

[^5]: M. Humt, J. Lee, R. Triebel, [Bayesian optimization meets Laplace approximation for robotic introspection](https://arxiv.org/abs/2010.16141), 2020, arXiv preprint arXiv:2010.16141.


