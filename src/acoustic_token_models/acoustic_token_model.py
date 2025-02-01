import einops
import tensorflow as tf

from losses import STFT, inverse_stft
from model_utils import (
    RMSNorm,
    Snake,
    SnakeBeta,
    ScalingLayer,
    WeightNormalization,
    PositionalEncoding,
)
from latent_transformer import LatentTransformer


def get_activation_layer(activation):
    if activation == "leakyrelu":
        return tf.keras.layers.LeakyReLU(0.1)
    elif activation == "snake":
        return Snake()
    elif activation == "snakebeta":
        return SnakeBeta()
    else:
        return tf.keras.layers.Activation(activation)


def conv1d(
    hidden_size,
    kernel_size,
    strides=1,
    dilation_rate=1,
    activation=None,
    dtype=None,
    name=None,
    use_max_norm=False,
    use_weight_norm=False,
    use_bias=True,
):
    constraint = None
    if use_max_norm:
        constraint = tf.keras.constraints.MaxNorm(max_value=2)
    if use_weight_norm:
        return WeightNormalization(
            tf.keras.layers.Conv1D(
                hidden_size,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding="same",
                dtype=dtype,
                activation=activation,
                name=name,
                kernel_constraint=constraint,
            ),
            data_init=False,
            dtype=dtype,
        )

    return tf.keras.layers.Conv1D(
        hidden_size,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        dtype=dtype,
        activation=activation,
        name=name,
        use_bias=use_bias,
        kernel_constraint=constraint,
    )


def conv1dtranspose(hidden_size, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return tf.keras.layers.Conv1DTranspose(
        hidden_size,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
    )


def dense(hidden_size, activation=None, use_bias=True):
    return tf.keras.layers.Dense(hidden_size, activation=activation, use_bias=use_bias)


class ConvNeXtModule(tf.keras.layers.Layer):
    def __init__(self, hidden_size, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

        self.depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size=7, padding="same")
        self.norm = RMSNorm()

        self.pointwise_conv1 = dense(hidden_size * 4, activation=get_activation_layer(activation))
        self.pointwise_conv2 = dense(hidden_size)
        self.gamma = self.add_weight(name="gamma", initializer="ones", trainable=True)

    def call(self, inputs, training=False):
        residual = inputs

        x = self.depthwise_conv(inputs)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.pointwise_conv2(x)
        x = self.gamma * x
        return x + residual


class ResidualConvModule2(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        kernel_size=3,
        dilations=(1,),
        activation="leakyrelu",
        use_max_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_layers = []

        for dilation in dilations:
            self.conv_layers.append(
                [
                    get_activation_layer(activation),
                    conv1d(
                        hidden_size,
                        kernel_size=kernel_size,
                        dilation_rate=dilation,
                        use_max_norm=use_max_norm,
                    ),
                    get_activation_layer(activation),
                    conv1d(
                        hidden_size,
                        kernel_size=kernel_size,
                        dilation_rate=1,
                        use_max_norm=use_max_norm,
                    ),
                ]
            )

    def call(self, inputs, training=False):
        for layer_set in self.conv_layers:
            residual = inputs
            for layer in layer_set:
                inputs = layer(inputs, training=training)

            inputs = residual + inputs
        return inputs


class MRFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        kernel_sizes=(
            3,
            7,
            11,
        ),
        dilations=(1, 3, 5),
        activation="leakyrelu",
        use_max_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer_blocks = []

        for kernel_size in kernel_sizes:
            self.layer_blocks += [
                ResidualConvModule2(
                    hidden_size,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    activation=activation,
                    use_max_norm=use_max_norm,
                ),
            ]

    def call(self, inputs, training=False):
        output = None
        for layer in self.layer_blocks:
            layer_out = layer(inputs, training=training)

            if output is None:
                output = layer_out
            else:
                output += layer_out

        output = output / len(self.layer_blocks)
        return output


class MISRModule(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        depth=3,
        kernel_size=11,
        dilations=(1, 3, 5),
        activation="leakyrelu",
        impl="fast",
        use_max_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.channel_expansion_conv = dense(hidden_size * depth)
        self.conv = ResidualConvModule2(
            hidden_size,
            kernel_size=kernel_size,
            dilations=dilations,
            activation=activation,
            use_max_norm=use_max_norm,
        )
        self.channel_reduction_conv = dense(hidden_size)
        self.impl = impl

    def _native_impl(self, inputs, training):
        inputs_list = tf.split(inputs, num_or_size_splits=self.depth, axis=-1)

        block_list = []
        for block in inputs_list:
            block = self.conv(block, training=training)
            block_list.append(block)

        inputs = tf.concat(block_list, axis=-1)
        return inputs

    def _fast_impl(self, inputs, training):
        inputs_split = tf.split(inputs, num_or_size_splits=self.depth, axis=-1)
        inputs = tf.concat(inputs_split, axis=0)
        inputs = self.conv(inputs, training=training)

        inputs_split = tf.split(inputs, num_or_size_splits=self.depth, axis=0)
        inputs = tf.concat(inputs_split, axis=-1)
        return inputs

    def call(self, inputs, training=False):
        inputs = self.channel_expansion_conv(inputs)

        if self.impl == "native":
            inputs = self._native_impl(inputs, training=training)
        elif self.impl == "fast":
            inputs = self._fast_impl(inputs, training=training)

        inputs = self.channel_reduction_conv(inputs)
        return inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor_1 = tf.keras.layers.Conv1D(
            hidden_size, kernel_size=4, strides=2, padding="same", use_bias=False
        )
        self.feature_extractor_2 = tf.keras.layers.Conv1D(
            hidden_size, kernel_size=4, strides=2, padding="same", use_bias=False
        )

        self.context_pos_enc = PositionalEncoding(hidden_size, dtype=tf.float32)
        self.context_network = LatentTransformer(hidden_size=hidden_size, num_heads=8, num_layers=8)

    def call(self, inputs, training=False):
        inputs = self.feature_extractor_1(inputs)
        inputs = self.feature_extractor_2(inputs)
        inputs = self.context_pos_enc(inputs, training=training)
        inputs = self.context_network(inputs, training=training)

        return inputs


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        strides,
        module_type="misr",
        decoder_activation="leakyrelu",
        num_layers=1,
        use_max_norm=False,
        use_scaling=False,
        scaling_loss_weight=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = [
            conv1d(hidden_size, kernel_size=7, use_max_norm=use_max_norm),
            get_activation_layer(decoder_activation),
        ]

        if use_scaling:
            self.layers += [ScalingLayer(loss_weight=scaling_loss_weight, dtype=tf.float32)]

        for stride in strides:
            self.layers += [
                conv1dtranspose(
                    hidden_size,
                    kernel_size=2 * stride,
                    strides=stride,
                ),
            ]

            for _ in range(num_layers):
                if use_scaling:
                    self.layers += [ScalingLayer(loss_weight=scaling_loss_weight, dtype=tf.float32)]
                if "misr" in module_type:
                    impl = "native"
                    if module_type == "misr_fast":
                        impl = "fast"
                    self.layers += [
                        MISRModule(
                            hidden_size,
                            activation=decoder_activation,
                            use_max_norm=use_max_norm,
                            impl=impl,
                        ),
                    ]
                elif module_type == "mrf":
                    self.layers += [
                        MRFModule(
                            hidden_size,
                            activation=decoder_activation,
                            use_max_norm=use_max_norm,
                        ),
                    ]
                elif module_type == "conv":
                    self.layers += [
                        ConvNeXtModule(hidden_size, activation=decoder_activation),
                    ]
                self.layers += [
                    get_activation_layer(decoder_activation),
                ]

            hidden_size /= 2
        if use_scaling:
            self.layers += [ScalingLayer(loss_weight=scaling_loss_weight, dtype=tf.float32)]

        self.layers += [dense(hidden_size * 4)]

    def call(self, inputs, training=False):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
            # tf.print("decoder:", layer.name, " min:", tf.reduce_min(inputs), "max:", tf.reduce_max(inputs))
        return inputs


class AcousticEncoderModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.encoder = Encoder(hidden_size=config.encoder_dim)
        self.embedding_conv = conv1d(config.encoder_embedding_dim, kernel_size=7)
        self.num_channels = config.num_channels

        self.stft_layer = STFT(
            frame_length=config.window_length,
            fft_length=config.n_fft,
            frame_step=config.hop_length,
            logscale=config.log_scale_input,
            dtype=tf.float32,
        )

    @classmethod
    def from_pretrain(
        cls,
        config,
        model_weight_path=None,
    ):
        model_input = tf.keras.layers.Input(shape=(None, config.num_channels))

        intermediate_model = cls(config=config)
        outputs = intermediate_model(model_input)
        model = tf.keras.Model(inputs=[model_input], outputs=outputs)

        if model_weight_path is not None:
            model.load_weights(model_weight_path)

        return model, intermediate_model

    def preprocess(self, signals):
        stft = self.stft_layer(signals, return_complex=True)
        stft = tf.transpose(stft, [0, 1, 3, 2])
        stft = einops.rearrange(stft, "b t c f -> b t (c f)")

        real = tf.math.real(stft)
        real = tf.where(tf.math.is_nan(real) | tf.math.is_inf(real), tf.zeros_like(real), real)

        imag = tf.math.imag(stft)
        imag = tf.where(tf.math.is_nan(imag) | tf.math.is_inf(imag), tf.zeros_like(imag), imag)
        return tf.concat([real, imag], axis=-1)

    def call(self, inputs, training=False):
        original_dtype = inputs.dtype

        inputs = self.preprocess(inputs)
        inputs = self.encoder(inputs, training=training)
        inputs = tf.cast(inputs, dtype=original_dtype)
        return inputs


class AcousticModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.encoder = AcousticEncoderModel(config)
        self.decoder = Decoder(
            hidden_size=config.decoder_dim,
            strides=config.decoder_strides,
            module_type=config.decoder_module_type,
            decoder_activation=config.decoder_activation,
            num_layers=config.decoder_num_layers,
            use_max_norm=config.decoder_use_max_norm,
            use_scaling=config.decoder_use_scaling,
            scaling_loss_weight=config.decoder_scaling_loss_weight,
        )

        if config.inverse_mode == "stft":
            self.last_conv = conv1d(
                (config.istft_n_fft // 2 + 1) * 2 * config.num_channels,
                kernel_size=7,
                dtype=tf.float32,
            )
        elif config.inverse_mode == "wave":
            self.last_conv = conv1d(config.num_channels, kernel_size=7, dtype=tf.float32)
        else:
            raise NotImplementedError("対応していないinvese_modeです")

        self.num_channels = config.num_channels
        self.istft_n_fft = config.istft_n_fft
        self.istft_hop_length = config.istft_hop_length
        self.istft_window_length = config.istft_window_length
        self.inverse_mode = config.inverse_mode
        self.sampling_rate = config.sampling_rate
        self.phase_activation = config.phase_activation

    @classmethod
    def from_pretrain(
        cls,
        config,
        model_weight_path=None,
    ):
        model_input = tf.keras.layers.Input(shape=(None, config.num_channels))

        intermediate_model = cls(config=config)
        outputs = intermediate_model(model_input)
        model = tf.keras.Model(inputs=[model_input], outputs=outputs)

        if model_weight_path is not None:
            model.load_weights(model_weight_path)

        return model, intermediate_model

    def inverse(self, inputs, use_padding=False, seq_len=None):
        inputs = self.last_conv(inputs)

        if self.inverse_mode == "wave":
            inputs = tf.clip_by_value(inputs, -1, 1)
            return inputs

        if use_padding:
            crop_length = tf.cast(tf.math.ceil(seq_len / self.istft_hop_length), dtype=tf.int32)
            padding_num = tf.maximum(tf.shape(inputs)[1] - crop_length, 0)
            inputs = tf.pad(inputs, [[0, 0], [0, padding_num], [0, 0]], mode="REFLECT")
            inputs = inputs[:, :crop_length]
        # 位相と振幅に分割
        split_inputs = tf.split(inputs, num_or_size_splits=2 * self.num_channels, axis=-1)

        spec = tf.math.exp(tf.stack(split_inputs[: self.num_channels], axis=-1))
        spec = tf.clip_by_value(spec, spec, 1e3)  # nan回避
        phase = tf.stack(split_inputs[self.num_channels :], axis=-1)
        if self.phase_activation == "sin":
            phase = tf.sin(phase)

        inputs = inverse_stft(
            spec,
            phase,
            frame_length=self.istft_window_length,
            frame_step=self.istft_hop_length,
            fft_length=self.istft_n_fft,
        )
        inputs = tf.clip_by_value(inputs, -1, 1)
        return inputs

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]

        inputs = self.encoder(inputs, training=training)
        inputs = self.decoder(inputs, training=training)
        inputs = self.inverse(inputs, use_padding=True, seq_len=seq_len)
        inputs = inputs[:, :seq_len]

        return inputs
