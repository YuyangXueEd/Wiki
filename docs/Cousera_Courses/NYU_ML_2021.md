# NYU Machine Learning

## Gradient Descent and Backpropagation algorithm

- Full (batch) gradient
	- $w \leftarrow w-\eta \frac{\partial \mathcal{L}(S, w)}{\partial w}$
- Stochastic Gradient (SGD)
	- Pick a $p$ in $0, \dots, P-1$, then update $w$
	- $w\leftarrow w-\eta \frac{\partial \mathcal{L}(x[p], y[p], w)}{\partial w}$
	- SGD exploits the redundancy in the samples
		- It goes faster than full gradient in most cases
		- In practice, we use mini-batches for parallelisation