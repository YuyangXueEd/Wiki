# Jax Tutorial

## NumPy API

来比较一下 `jax` 中的 `numpy` 和原生 `numpy` 的不同之处。

```py

import jax
from jax import numpy as jnp, random

import numpy as np
```

`jax.numpy` 是类似 `NumPy` 的接口，并且我们也要使用 `jax.random`来生成数据：

```py
m = jnp.ones((4, 4)) # 4 by 4 matrix filled with 1
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
			   [5.0, 6.0, 7.0, 8.0]]) # 2 by 4 array
```

JAX中的数组表示为 `DeviceArray` 实例，并且与数组所在的位置无关（如CPU或者GPU）。如果您没有GPU或者TPU，JAX就会提出警告。

```
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

DeviceArray([[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]], dtype=float32)
```

和 `NumPy` 中一样，使用 `jnp` 做乘法操作：

```py

jnp.dot(n, m).block_until_ready()
```

结果如下：

```
DeviceArray([[10., 10., 10., 10.],
             [26., 26., 26., 26.]], dtype=float32)
```

由于JAX中默认的异步执行， `DeviceArray` 实际上是一个未计算的量。由于这个原因，Python调用可能会在实际计算结束之前返回，因此我们使用 `block_until_ready()` 方法来确保我们返回最终结果。

当然，JAX完全兼容 Numpy，因此我们可以直接在两个库上进行大部分操作。但如果我们要将数据移到GPU或TPU上时，我们可以直接构造一个 `DeviceArray` 或者在已有的 Numpy 数组上使用 `jax.device_put`, 以下为例：

```py

x = np.random.normal(size=(4, 4))
x = jax.device_put(x)
x
```

相对的，如果你想从 JAX 数组转换到 Numpy，也可以直接使用 Numpy 的接口：

```py

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
			   [5.0, 6.0, 7.0, 8.0]])
np.array(x)
```

结果为：

```
array([[1., 2., 3., 4.],
       [5., 6., 7., 8.]], dtype=float32)
```

## （不）可变性

JAX 本质上是函数式的，这样导致 JAX 数组是不可变的。 这意味着没有 inplace 操作和切片分配。 也就是说，函数不应使用全局状态获取输入或产生输出。

```py

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
# x[0, 0] = 3.0 would fail
updated = x.at[0, 0].set(3.0) # copy and edit
print("x: \n", x)
print("updated: \n", updated)
```

```
x: 
 [[1. 2. 3. 4.]
 [5. 6. 7. 8.]]
updated: 
 [[3. 2. 3. 4.]
 [5. 6. 7. 8.]]
```

## 管理随机性

在 JAX 中我们可以手动管理随机性。简而言之，我们需要明确管理 PRNG（伪随机数生成器）及其状态。 在 JAX 的 PRNG 中，状态表示为一对称为密钥的两个 `unsigned int32`（这两个 `unsigned int32` 没有特殊含义——它只是表示 `uint64` 的一种方式）。

```py

key = random.PRNGKey(0)
key
```

```
DeviceArray([0, 0], dtype=uint32)
```

如果你多次使用这个密钥，你每次都会得到相同的“随机”输出。 

```py

for i in range(3):
    print("Printing the random number using key: ", key, " gives: ", random.normal(key,shape=(1,))) 
    # Boringly not that random since we use the same key
```

这样只会让结果一模一样：

```
Printing the random number using key:  [0 0]  gives:  [-0.20584226]
Printing the random number using key:  [0 0]  gives:  [-0.20584226]
Printing the random number using key:  [0 0]  gives:  [-0.20584226]
```

要在序列中生成更多条目，您需要“拆分” PRNG，从而生成一对新的密钥。

```py

print("old key", key, "--> normal", random.normal(key, shape=(1, )))
key, subkey = random.split(key)
print("    \---SPLIT --> new key   ", key, "--> normal", random.normal(key, shape=(1,)) )
print("             \--> new subkey", subkey, "--> normal", random.normal(subkey, shape=(1,)) )
```

这样就更新了随机数：

```
old key [0 0] --> normal [-0.20584226]
    \---SPLIT --> new key    [4146024105  967050713] --> normal [0.14389051]
             \--> new subkey [2718843009 1272950319] --> normal [-1.2515389]
```

同样的我们也可以一次性生成多个密钥：

```py

key, *subkeys = random.split(key, 4)
key, subkeys
```

```
(DeviceArray([3306097435, 3899823266], dtype=uint32),
 [DeviceArray([147607341, 367236428], dtype=uint32),
  DeviceArray([2280136339, 1907318301], dtype=uint32),
  DeviceArray([ 781391491, 1939998335], dtype=uint32)])
```

这么做的目的是对于我们期望的随机行为的可重复性和可靠性的重视。

## 梯度和自动微分

尽管从理论上讲，VJP（Vector-Jacobian product - reverse autodiff）和 JVP（Jacobian-Vector product - forward-mode autodiff）是相似的——它们计算 Jacobian 和向量的乘积——它们的不同之处在于计算复杂度。简而言之，当您有大量参数（因此是宽矩阵）时，JVP 在计算上的效率低于 VJP，相反，当雅可比矩阵是高矩阵时，JVP 效率更高。

### 梯度

JAX 为函数中的梯度和自动微分提供一流的支持。如果我们考虑一个简单的函数：

$$f(x)=\frac12 x^Tx,\ f: \mathbb{R}^n\rightarrow \mathbb{R}$$

求得的梯度为：

$$\triangledown f(x)=x$$

``` py
key = random.PRNGKey(0)
def f(x):
	return jnp.dot(x.T, x) / 2.0

v = jnp.ones((4, ))
```

JAX 将梯度计算为使用 `jax.grad` 作用于函数的运算符。 请注意，这仅适用于**标量值**函数。

让我们取 `f` 的梯度并确保它与恒等映射匹配:

```py

v = random.normal(key, (4, ))
print("Original v:")
print(v)
print("Gradient of f taken at point v")
print(jax.grad(f)(v)) # should be equal to v !
```

```
Original v:
[ 1.8160863  -0.75488514  0.33988908 -0.53483534]
Gradient of f taken at point v
[ 1.8160863  -0.75488514  0.33988908 -0.53483534]
```

如前所述，`jax.grad` 仅适用于标量值函数。 JAX 还可以处理一般的向量值函数。 最有用的原语是 Jacobian-Vector 乘积 - `jax.jvp` - 和 Vector-Jacobian 乘积 - `jax.vjp`。

## 使用 `jit` 和 `ops` 矢量化加速代码

### Jit

JAX 在后台使用 XLA 编译器，使您能够 jit 编译代码以使其更快、更高效。 这就是 `@jit` 修饰符的目的。

如果在不使用 `@jit` 的情况下计算 `selu()` 函数：
```py

def selu(x, alpha=1.67, lmbda=1.05):
	return lambda * jnp.where(x >0, x, alpha * jnp.exp(x) - alpha)

v = random.normal(key, (1000000))
%timeit selu(v).block_until_ready()
```

```
3.2 ms ± 13.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

现在我们加上 `jit` （这里是函数，也可以使用修饰符）来加速：

```py

selu_jit = jax.jit(selu)
%timeit selu_jit(v).block_until_ready()
```

```
860 µs ± 37.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

### 向量化

JAX 使您能够编写适用于单个样例的代码，然后对其进行矢量化以透明地管理批处理维度。

```py

mat = random.normal(key, (15, 10))
batched_x = random.normal(key, (5, 10)) # batch size on axis 0
single = random.normal(key, (10, ))

def apply_matrix(v):
	return jnp.dot(mat, v)

print("Single apply shape: ", apply_matrix(single).shape)
print("Batched example shape: ", jax.vmap(apply_matrix)(batched_x).shape)
```
```
Single apply shape:  (15,)
Batched example shape:  (5, 15)
```

## 案例：线性回归

