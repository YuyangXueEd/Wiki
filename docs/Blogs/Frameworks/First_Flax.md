# Flax初探

以下以一个完整的 `MNIST` 例子作为参考，介绍 `Flax` ，一个基于`JAX` 的框架的用法，其代码风格基于谷歌。

## `train`

这部分代码用于执行MNIST的训练和评估循环。

### Imports 

```py
# train file
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import  numpy as np
import optax 
import tensorflow_datasets as tfds
```

`absl` 库是由谷歌从自己的Python代码基地搜集而来的，可以用于简单的应用创建，分布式的命令行标志系统（类似`argparse`），用户自定义的记录模块（类似`tensorflow`中的`FLAGS`），以及方便的测试。在`main`函数中我们还可以看到更详细的使用，在`train`中暂时只使用了`logging`。

`flax`是我们主要使用的框架，`linen` 为`flax`的模块系统，与 `torch.nn` 类似，是神经网络库。

`flax.metrics`中内置了`tensorboard`，与`tensorflow`中的`tensorboard`用法基本一致。

其余的引用我们在使用的时候继续介绍。

### Define network

首先先来看网络模型定义，与`tensorflow 2.0`和`PyTorch`类似，由于是线性模型，可以很方便地直接在`__call__`方法内直接堆叠网络层，并使用`@nn.compact`修饰符来指示。代码如下：

```py
## a simple CNN model
class CNN(nn.Module): #same as PyTorch
	@nn.compact
	def __call__(self, x):
		x = nn.Conv(features=32, kernel_size=(3, 3))(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = nn.Conv(features=64, kernel_size=(3, 3))(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = x.reshape((x.shape[0], -1)) # flatten
		x = nn.Dense(features=256)(x)
		x = nn.relu(x)
		x = nn.Dense(features=10)(x)
		return x
```

与`PyTorch`不同的是，`flax`不需要将其分为`__init__`和`forward`两个部分，而是直接用一个`__call__`来解决问题，更直接一点。

### Define loss and metric

接下来我们试着给这个模型上应用每一步的梯度计算和损失函数计算，同时评估其准确率。在函数定义前应用`jax`的`@jit`修饰器来跟踪每个变量，并使用XLA将其即时编译为融合设备操作，从而在硬件加速器上更快、更高效地执行：

- 使用 `Module.apply` 方法评估给定参数和一批输入图像的神经网络。
- 计算 `cross_entropy_loss` 损失函数。
- 使用 `jax.value_and_grad` 评估损失函数及其梯度。
- 将梯度 `pytree` 应用到优化器已更新模型的参数。
- 计算精确度指标

```py
"""Computes gradients, loss and accuracy for a single batch."""
@jax.jit
def apply_model(state, images, labels):
	def loss_fn(params):
		logits = state.apply_fn({'params': params}, images)
		one_hot = jax.nn.one_hot(labels, 10)
		loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
		return loss, logits
	grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
	(loss, logits), grads = grad_fn(state.params)
	accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
	return grads, loss, accuracy
```

首先我们定义了损失函数 `loss_fn`，其参数为 `params`，由于 `train state` 中需要，我们之后再展开说。这里简单地使用了`optax.softmax_cross_entropy()`，但是这个函数需要成对的 `logits` 和 `labels`，且拥有 `[batch, num_classes]` 的形状。同时，我们将要从 TFDS 中读入的 `MNIST` 数据集中的标签是整形数，所以需要使用内置的 `jax.nn.onehot` 来转变编码方式。之后，函数返回一个准备优化的简单标量值，因此我们取 `optax` 损失函数返回的向量的平均值。

更简单直观的理解，输入 `image` 通过网络 `state.apply_fn` 得到输出 `logits`， `label` 需要通过重新编码转换为 `one_hot`，最后讲这两者通过计算得到 `loss`。

这里的 `optax` 是 `JAX` 的梯度处理和优化库，类似 `PyTorch` 中的 `torch.optim`，允许在一行代码中实现许多标准优化器。

`jax.value_and_grad` 函数创建一个评估输入和输入梯度的函数， `has_aux` 提示是否返回一对，其中第二个是辅助元素。在这个例子中，`grad_fn` 通过网路参数得到 `(loss, logits)` 和其辅助元素 `grads`。

### Loading data

这里使用`tensorflow_datasets` 来加载数据。

```py
"""Load MNIST train and test datasets into memory."""
def get_datasets():
	ds_builder = tfds.builder('mnist', data_dir='tensorflow_datasets')
	ds_builder.download_and_prepare()
	train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
	test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
	train_ds['image'] = jnp.float32(train_ds['image']) / 255.
	test_ds['image'] = jnp.float32(test_ds['image']) / 255.

	return train_ds, test_ds
```

### Create train state

`Flax` 中的一个常见模式是创建一个表示整个训练状态的单个数据类，包括步数、参数和优化器状态。此外，将优化器和模型添加到此状态的优点是我们只需将单个参数传递给诸如 `train_step()` 之类的函数。

`Flax` 提供类 `flax.training.train_state.TrainState` 服务于大多数基本永利，通常会对其进行子类化以添加更多要跟踪的数据，但在此示例中，我们可以在不进行任何修改的情况下使用它。

```
"""Create initial TrainState"""
def create_train_state(rng, config):
	cnn = CNN()
	params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
	tx = optax.sgd(config.learning_rate, config.momentum)

	return train_state.TrainState.create(
		apply_fn=cnn.apply, params=params, tx=tx)
```

这里，`apply_fn` 一般设定为 `model.apply()`，`tx` 一般为 `Optax` 的梯度变换。

### Training Step

以上是单步训练的内容，而一整个 epoch 的内容包括了：

- 使用以 `PRNGKey` 作为参数的 `jax.random.permutation` 在每个epoch之前对训练数据来随机
- 为每个 `batch` 运行优化步骤
- 返回具有更新参数以及训练损失和准确度指标的优化器。

```py
"""Train for a single epoch"""
@jax.jit
def update_model(state, grads):
	return state.apply_gradients(grads=grads)

def train_epoch(state, train_ds, batch_size, rng):
	train_ds_size = len(train_ds['image'])
	steps_per_epoch = train_ds_size // batch_size

	perms = jax.random.permutation(rng, len(train_ds['image']))
	perms = perms[:steps_per_epoch * batch_size] # skip incomplete batch
	perms = perms.reshape((steps_per_epoch, batch_size))

	epoch_loss = []
	epoch_accuracy = []

	for perm in perms:
		batch_images = train_ds['image'][perm, ...]
		batch_labels = train_ds['label'][perm, ...]
		grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
		state = update_model(state, grads)
		epoch_loss.append(loss)
		epoch_accuracy.append(accuracy)

	train_loss = np.mean(epoch_loss)
	train_accuracy = np.mean(epoch_accuracy)

	return state, train_loss, train_accuracy
```

### Train and Evaluate

```py
def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState：
	"""
	Execute model training and evaluation loop
	Args:
		config: Hyperparameter configuration for training and evaluation
		workdir: Directory where the tensorboard summaries are written to
	Returns:
		The train state
	"""

	train_ds, test_ds = get_datasets()
	rng = jax.random.PRNGKey(0)

	summary_writer = tensorboard.SummaryWriter(workdir)
	summary_writer.hparams(dict(config))
	
	rng, init_rng = jax.random.split(rng)
	state = create_train_state(init_rng, config)

	for epoch in range(1, config.num_epochs + 1):
		rng, input_rng = jax.random.split(rng)
		state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
		_, test_loss, test_accuracy = apply_model(state, test_ds['image'], test_ds['label'])

		logging.info( 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
	    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

	summary_writer.flush()
	return state
```

这是汇总的训练和评估循环，这里指定了输出的 `checkpoint` 文件位置，并且指定了随机数种子。在epoch循环中，使用 `split` 得到新的随机数，用之前定义的 `train_epoch` 和 `apply_model` 来训练和计算梯度和损失函数。


## `main` 

### import

```py
# main file
from absl import app
from absl import flags 
from absl import logging
from clu import platform

import jax
from ml_collections imioprt config_flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
	'config',
	None,
	'File path to the training hyperparameter configuration.',
	lock_config=True
)
```

在 `main` 文件中，`absl` 使用 `app` 来指定该程序，使用 `flags` 来接收超参数。

`clu` 库是通用循环实用程序，包含用于编写ML训练循环的常用功能。

### main function

```py

def main(argv):
	if len(argv) > 1:
		raise app.UsageError('Too many command-line arguments.')
	# Hide GPUs from Tensorflow
	tf.config.experimental.set_visible_devices([], 'GPU')

	logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
	logging.info('JAX local devices: %r', jax.local_devices())

	# Add a note so that we can t3ell which task is which JAX host
	platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
	f'process_count: {jax.process_count()}')
	platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir')

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
	flags.mark_flags_as_required(['config', 'workdir'])
	app.run(main)
```

这里设置了将GPU对Tensorflow隐藏，否则TF可能会保留现存并使其对JAX不可用。

