import tensorflow as tf
from model_utils import GLU, RMSNorm


class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self, hidden_size, expansion_rate=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.dense_1 = tf.keras.layers.Dense(
            hidden_size * expansion_rate, use_bias=False
        )
        self.activation_1 = tf.keras.layers.Activation("swish")
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dense_2 = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        inputs = self.norm_1(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.activation_1(inputs)

        inputs = self.dropout_1(inputs, training=training)
        inputs = self.dense_2(inputs)
        inputs = self.dropout_2(inputs, training=training)
        return inputs

    def get_config(self):
        config = super().get_config()
        return config


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self, hidden_size, expansion_rate=2, kernel_size=32, dropout=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.pointwise_conv_1 = tf.keras.layers.Dense(
            hidden_size * expansion_rate, use_bias=False
        )
        self.activation_1 = GLU()
        self.depth_conv_1 = tf.keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size, use_bias=False, padding="same"
        )
        self.norm_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.Activation("swish")
        self.pointwise_conv_2 = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        inputs = self.norm_1(inputs)
        inputs = self.pointwise_conv_1(inputs)
        inputs = self.activation_1(inputs)

        inputs = self.depth_conv_1(inputs)
        inputs = self.norm_2(inputs)
        inputs = self.activation_2(inputs)

        inputs = self.pointwise_conv_2(inputs)
        inputs = self.dropout_1(inputs, training=training)
        return inputs

    def get_config(self):
        config = super().get_config()
        return config


class LatentAttention(tf.keras.layers.Layer):
    def __init__(self, latent_dim, query_dim, key_dim=None, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads

        if self.key_dim is None:
            self.key_dim = self.query_dim

        self.wq = self.add_weight(
            name="wq",
            shape=(self.query_dim, self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.wk = self.add_weight(
            name="wk",
            shape=(self.key_dim, self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.wv = self.add_weight(
            name="wv",
            shape=(self.key_dim, self.query_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.out_proj = self.add_weight(
            name="out_proj",
            shape=(self.query_dim, self.query_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, query, key, value=None, training=False):
        if value is None:
            value = key

        batch = tf.shape(query)[0]
        time = tf.shape(query)[1]

        Q = tf.einsum("d l, b t d -> t b l", self.wq, query)
        K = tf.einsum("d l, b t d -> t b l", self.wk, key)
        V = tf.einsum("d m, b t d -> t b m", self.wv, value)

        Q = tf.reshape(Q, [tf.shape(query)[1], tf.shape(query)[0], self.num_heads, -1])
        K = tf.reshape(K, [tf.shape(key)[1], tf.shape(key)[0], self.num_heads, -1])
        V = tf.reshape(V, [tf.shape(value)[1], tf.shape(value)[0], self.num_heads, -1])

        Qs = tf.math.softmax(Q, axis=-1)
        maxi = tf.reduce_max(K, axis=0, keepdims=True)
        K = tf.exp(K - maxi)

        Kv = tf.einsum("t b h l, t b h d -> b h l d", K, V)
        K = tf.reduce_sum(K, axis=0)
        Kv = tf.einsum("b h l, b h l d -> b h l d", 1 / K, Kv)
        y = tf.einsum("t b h l, b h l d -> b t h d", Qs, Kv)
        y = tf.reshape(y, [batch, time, -1])
        return y @ self.out_proj

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


class LatentTransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        latent_dim,
        num_heads,
        layer_norm_eps=1e-5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout

        self.attention_layer = LatentAttention(
            latent_dim, self.hidden_size, num_heads=self.num_heads
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_size * 2),
                GLU(),
                tf.keras.layers.Dense(hidden_size),
            ]
        )
        self.layernorm1 = RMSNorm()
        self.layernorm2 = RMSNorm()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        residual = inputs
        inputs = self.layernorm1(inputs)
        inputs = self.attention_layer(inputs, inputs, training=training)
        inputs = self.dropout1(inputs, training=training)
        inputs = inputs + residual

        residual = inputs
        inputs = self.layernorm2(inputs, training=training)
        inputs = self.ffn(inputs)
        inputs = self.dropout2(inputs)
        inputs = inputs + residual
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout,
            }
        )

        return config


class LatentTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_heads=8,
        layer_norm_eps=1e-5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = [
            LatentTransformerLayer(
                hidden_size=hidden_size,
                latent_dim=hidden_size,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

    def call(
        self,
        inputs,
        training=False,
        return_intermediate_layer=False,
    ):
        layer_outputs = []
        for layer in self.layers:
            inputs = layer(inputs, training=training)
            layer_outputs.append(inputs)

        if return_intermediate_layer:
            return inputs, layer_outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        return config
