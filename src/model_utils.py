import tensorflow as tf
import numpy as np


def cosine_similarity(a, b):
    a_normalized = tf.nn.l2_normalize(a, axis=-1)
    b_normalized = tf.nn.l2_normalize(b, axis=-1)
    cosine_sim = a_normalized * b_normalized
    return tf.reduce_sum(cosine_sim, axis=-1)


def cosine_similarity_matmul(a, b, transpose_b=True):
    a_normalized = tf.nn.l2_normalize(a, axis=-1)
    if transpose_b:
        b_normalized = tf.nn.l2_normalize(b, axis=-1)
    else:
        b_normalized = tf.nn.l2_normalize(b, axis=-2)

    cosine_similarities = tf.matmul(a_normalized, b_normalized, transpose_b=transpose_b)
    return cosine_similarities


def create_delay_pattern(embeddings, k):
    delayed_embeddings = []

    for i in range(k):
        codebook_embeddings = embeddings[i]

        padding = [[0, 0], [i, k - i - 1], [0, 0]]
        padded_embedding = tf.pad(
            codebook_embeddings, paddings=padding, mode="CONSTANT", constant_values=0
        )
        delayed_embeddings.append(padded_embedding)

    concatenated_embeddings = tf.concat(delayed_embeddings, axis=-1)
    return concatenated_embeddings


def reverse_delay_pattern(delayed_embeddings, k):
    _, _, embedding_dim = delayed_embeddings.get_shape().as_list()
    original_embeddings = []
    for i in range(k):
        start = int(i * (embedding_dim / k))
        end = int((i + 1) * (embedding_dim / k))

        if i < k - 1:
            embeddings = delayed_embeddings[:, i : -(k - i - 1), start:end]
        else:
            embeddings = delayed_embeddings[:, i:, start:end]

        original_embeddings.append(embeddings)
    return original_embeddings


@tf.custom_gradient
def grad_multiply(x, scale):
    scale = tf.cast(scale, dtype=x.dtype)

    def grad(dy):
        return dy * scale, None

    return x, grad


def minmax(inputs, axis=-1):
    min_value = tf.reduce_min(inputs, axis=axis, keepdims=True)
    max_value = tf.reduce_max(inputs, axis=axis, keepdims=True)
    return (inputs - min_value) / (max_value - min_value)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, feature_size, max_shift_size=100, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_shift_size = max_shift_size
        self.feature_size = feature_size

    def __call__(self, inputs, mask=None, training=None):
        def positional_encoding(length, depth):
            depth = depth / 2

            positions = tf.range(length)[:, tf.newaxis]  # (length, 1)
            depths = tf.range(depth)[tf.newaxis, :]  # (1, depth)

            angle_rates = 1 / tf.pow(10000.0, depths)  # (1, depth)
            angle_rads = (
                tf.cast(positions, dtype=tf.float32) * angle_rates
            )  # (length, depth)

            pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
            return pos_encoding

        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs *= tf.math.sqrt(tf.cast(self.feature_size, tf.float32))

        if training is False:
            return (
                inputs
                + tf.cast(
                    positional_encoding(tf.shape(inputs)[-2], self.feature_size),
                    dtype=inputs.dtype,
                )[tf.newaxis, ...]
            )

        position_offset = np.random.randint(0, self.max_shift_size)
        return inputs + tf.cast(
            positional_encoding(
                tf.shape(inputs)[-2] + position_offset, self.feature_size
            )[tf.newaxis, ...][:, position_offset:],
            dtype=inputs.dtype,
        )

    def get_config(self):
        config = {
            "feature_size": self.feature_size,
            "max_shift_size": self.max_shift_size,
        }
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/layers/wrappers.py
class WeightNormalization(tf.keras.layers.Wrapper):
    """Performs weight normalization.

    This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.

    See [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868).

    Wrap `tf.keras.layers.Conv2D`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = WeightNormalization(tf.keras.layers.Conv2D(2, 2), data_init=False)
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])

    Wrap `tf.keras.layers.Dense`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = WeightNormalization(tf.keras.layers.Dense(10), data_init=False)
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])

    Args:
      layer: A `tf.keras.layers.Layer` instance.
      data_init: If `True` use data dependent variable initialization.
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights.
      NotImplementedError: If `data_init` is True and running graph execution.
    """

    def __init__(self, layer: tf.keras.layers, data_init: bool = True, **kwargs):
        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            trainable=True,
            experimental_autocast=False,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config["config"]["trainable"] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        inputs = tf.cast(inputs, dtype=self.g.dtype)

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            kernel = (
                tf.nn.l2_normalize(
                    tf.cast(self.v, dtype=g.dtype), axis=self.kernel_norm_axes
                )
                * g
            )
            kernel = tf.cast(kernel, dtype=self.layer.compute_dtype)

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.

        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized, False, message="The layer has been initialized."
                )
            ]
        ):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            v_flat = tf.reshape(
                tf.cast(self.v, dtype=self.g.dtype), [-1, self.layer_depth]
            )
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, "bias") and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name="recurrent_kernel" if self.is_rnn else "kernel",
        )

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer


class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, loss_weight=0.01, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.loss_weight = loss_weight

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(1),
            trainable=True,
            experimental_autocast=False,
        )

    def call(self, inputs, training=None):
        inputs = inputs * self.scale

        l2_norm = tf.reduce_mean(tf.reduce_sum(tf.square(inputs), axis=-1))
        self.add_loss(l2_norm * self.loss_weight)
        return inputs


class DropPath(tf.keras.layers.Layer):
    def __init__(self, rate=0.5, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        def dropped_inputs(inputs):
            input_shape = tf.shape(inputs)
            batch_size = input_shape[0]
            random_tensor = (1 - self.rate) + tf.random.uniform(
                [
                    batch_size,
                ],
                dtype=inputs.dtype,
            )
            drop_shape = (batch_size,) + (1,) * (len(input_shape) - 1)
            binary_tensor = tf.reshape(tf.floor(random_tensor), drop_shape)
            output = tf.divide(inputs, 1 - self.rate) * binary_tensor
            return output

        if not training:
            return inputs

        return dropped_inputs(inputs)

    def get_config(self):
        config = super(DropPath, self).get_config()
        config.update({"rate": self.rate, "seed": self.seed})
        return config


class Snake(tf.keras.layers.Layer):
    def __init__(self, alpha_logscale=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha_logscale = alpha_logscale

    def build(self, input_shape):
        initial_value = 1
        if self.alpha_logscale:
            initial_value = 0

        self.alpha = self.add_weight(
            name="alpha",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(initial_value),
            trainable=True,
            experimental_autocast=False,
        )

    def call(self, x):
        original_dtype = x.dtype

        x = tf.cast(x, self.alpha.dtype)
        alpha = self.alpha
        alpha = tf.reshape(alpha, [1, 1, tf.shape(alpha)[0]])
        if self.alpha_logscale:
            alpha = tf.exp(alpha)
        x = x + (1.0 / (alpha + 1e-9)) * tf.pow(tf.sin(x * alpha), 2)
        return tf.cast(x, dtype=original_dtype)


class SnakeBeta(tf.keras.layers.Layer):
    def __init__(self, alpha_logscale=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha_logscale = alpha_logscale

    def build(self, input_shape):
        initial_value = 1
        if self.alpha_logscale:
            initial_value = 0

        self.alpha = self.add_weight(
            name="alpha",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(initial_value),
            trainable=True,
            experimental_autocast=False,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(initial_value),
            trainable=True,
            experimental_autocast=False,
        )

    def call(self, x):
        original_dtype = x.dtype

        x = tf.cast(x, self.alpha.dtype)
        alpha = self.alpha
        alpha = tf.reshape(alpha, [1, 1, tf.shape(alpha)[0]])
        beta = self.beta
        beta = tf.reshape(beta, [1, 1, tf.shape(beta)[0]])
        if self.alpha_logscale:
            alpha = tf.exp(alpha)
            beta = tf.exp(beta)
        x = x + (1.0 / (beta + 1e-9)) * tf.pow(tf.sin(x * alpha), 2)
        return tf.cast(x, dtype=original_dtype)


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.new_std = self.add_weight(
            name="new_std",
            shape=[
                input_shape[self.axis],
            ],
            initializer=tf.constant_initializer(1),
            trainable=True,
            experimental_autocast=False,
        )

    def call(self, x):
        original_dtype = x.dtype

        x = tf.cast(x, self.new_std.dtype)
        ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        norm_inputs = x * tf.math.rsqrt(ms + self.epsilon)
        return tf.cast(norm_inputs * self.new_std, dtype=original_dtype)

    def get_config(self):
        config = {"axis": self.axis, "epsilon": self.epsilon}
        base_config = super(RMSNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(0),
            trainable=True,
            experimental_autocast=False,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=[
                input_shape[-1],
            ],
            initializer=tf.constant_initializer(0),
            trainable=True,
            experimental_autocast=False,
        )

    def call(self, x):
        original_dtype = x.dtype

        x = tf.cast(x, self.gamma.dtype)
        Gx = tf.norm(x, ord="euclidean", axis=1, keepdims=True)
        Nx = Gx / (tf.reduce_mean(Gx, axis=-1, keepdims=True) + self.epsilon)
        return tf.cast(self.gamma * (x * Nx) + self.beta + x, dtype=original_dtype)

    def get_config(self):
        config = {"epsilon": self.epsilon}
        base_config = super(GRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(GlobalNorm, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        original_dtypes = inputs.dtype

        inputs = tf.cast(inputs, dtype=tf.float32)
        squared_mean = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        normalized_inputs = inputs / tf.sqrt(squared_mean + self.epsilon)
        return tf.cast(normalized_inputs, dtype=original_dtypes)


class GLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x


class ReGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(ReGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.nn.relu(gate)
        return tf.multiply(out, gate)

    def get_config(self):
        config = {"bias": self.bias, "dim": self.dim}
        base_config = super(ReGLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedAverageLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.avg_weights = self.add_weight(
            shape=(len(input_shape),),
            initializer="ones",
            trainable=True,
            name="avg_weights",
        )

    def call(self, inputs):
        weighted_sum = tf.add_n(
            [self.avg_weights[i] * inputs[i] for i in range(self.avg_weights.shape[0])]
        )
        return weighted_sum / tf.reduce_sum(self.avg_weights)
