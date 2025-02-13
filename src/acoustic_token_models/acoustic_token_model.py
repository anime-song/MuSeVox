import tensorflow as tf

from losses import STFT, inverse_stft_complex
from model_utils import (
    RMSNorm,
    Snake,
    SnakeBeta,
    WeightNormalization,
    PositionalEncoding,
)
from latent_transformer import LatentTransformer
from acoustic_token_models.SCNet import FusionLayer, SDBlock, SULayer


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


class DualPathBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, chunks_length, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.time_pos_enc = PositionalEncoding(hidden_size, max_shift_size=40, dtype=tf.float32)
        self.time_transformer = LatentTransformer(hidden_size, num_layers=1)
        self.time_linear = tf.keras.layers.Dense(hidden_size)
        self.time_norm = RMSNorm()

        self.freq_proj = tf.keras.layers.Dense(hidden_size)
        self.freq_pos_enc = PositionalEncoding(hidden_size, max_shift_size=40, dtype=tf.float32)
        self.freq_transformer = LatentTransformer(hidden_size, num_layers=1)
        self.freq_linear = tf.keras.layers.Dense(hidden_size)
        self.freq_norm = RMSNorm()

    def call(self, inputs, training=False):
        # inputs: (batch * segment_num, chunks_length, freq, num_channels)
        batch_size = tf.shape(inputs)[0]
        chunks_length = tf.shape(inputs)[1]
        freq = tf.shape(inputs)[2]
        num_channels = tf.shape(inputs)[3]

        # freq
        residual = inputs
        inputs = tf.reshape(inputs, [batch_size * chunks_length, freq, num_channels])
        inputs = self.freq_proj(inputs)
        inputs = self.freq_pos_enc(inputs, training=training)
        inputs = self.freq_transformer(inputs, training=training)
        inputs = self.freq_linear(inputs)
        inputs = self.freq_norm(inputs)
        inputs = tf.reshape(inputs, [batch_size, chunks_length, freq, num_channels])
        inputs = inputs + residual

        # time
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        residual = inputs
        inputs = tf.reshape(inputs, [batch_size * freq, chunks_length, num_channels])
        inputs = self.time_pos_enc(inputs, training=training)
        inputs = self.time_transformer(inputs, training=training)
        inputs = self.time_linear(inputs)
        inputs = self.time_norm(inputs)
        inputs = tf.reshape(inputs, [batch_size, freq, chunks_length, num_channels])
        inputs = inputs + residual

        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        return inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size_list=[4, 32, 64, 128],
        band_SR=[0.175, 0.392, 0.433],
        band_strides=[1, 4, 16],
        band_kernels=[3, 4, 16],
        conv_depths=[3, 2, 1],
        compress=4,
        conv_kernel=3,
        num_dual_path_blocks=4,
        chunks_length=40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunks_length = chunks_length

        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {"SR": band_SR[i], "stride": band_strides[i], "kernel": band_kernels[i]}
            for i in range(len(band_keys))
        }
        self.conv_config = {"compress": compress, "kernel": conv_kernel}

        self.sd_blocks = []
        for index in range(len(hidden_size_list) - 1):
            self.sd_blocks.append(
                SDBlock(
                    channels_out=hidden_size_list[index + 1],
                    band_configs=self.band_configs,
                    conv_config=self.conv_config,
                    depths=conv_depths,
                )
            )

        self.dual_path_layers = [
            DualPathBlock(hidden_size_list[-1], chunks_length) for _ in range(num_dual_path_blocks)
        ]

    def call(self, inputs, training=False):
        skip_list = []
        lengths_list = []
        original_lengths_list = []

        for sd_layer in self.sd_blocks:
            inputs, skip, lengths, original_lengths = sd_layer(inputs, training=training)
            skip_list.append(skip)
            lengths_list.append(lengths)
            original_lengths_list.append(original_lengths)

        for layer in self.dual_path_layers:
            inputs = layer(inputs, training=training)
        return inputs, skip_list[::-1], lengths_list[::-1], original_lengths_list[::-1]


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size_list=[4, 32, 64, 128],
        band_SR=[0.175, 0.392, 0.433],
        band_strides=[1, 4, 16],
        band_kernels=[3, 4, 16],
        num_separate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {"SR": band_SR[i], "stride": band_strides[i], "kernel": band_kernels[i]}
            for i in range(len(band_keys))
        }
        self.su_layers = []
        for index in range(len(hidden_size_list) - 1):
            self.su_layers.append(
                (
                    FusionLayer(channels=hidden_size_list[index + 1]),
                    SULayer(
                        channels_out=hidden_size_list[index] if index != 0 else hidden_size_list[index] * num_separate,
                        band_configs=self.band_configs,
                    ),
                )
            )

        self.su_layers = self.su_layers[::-1]

    def call(self, inputs, skip_list, lengths_list, original_lengths_list, training=False):
        for (fusion_layer, su_layer), skip, lengths, original_lengths in zip(
            self.su_layers, skip_list, lengths_list, original_lengths_list
        ):
            inputs = fusion_layer(inputs, skip=skip, training=training)
            inputs = su_layer(inputs, lengths=lengths, original_lengths=original_lengths, training=training)

        return inputs


class AcousticEncoderModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.encoder = Encoder(
            hidden_size_list=config.hidden_size_list,
            band_SR=config.band_SR,
            band_strides=config.band_strides,
            band_kernels=config.band_kernels,
            num_dual_path_blocks=config.num_dual_path_blocks,
        )
        self.num_channels = config.num_channels

        self.stft_layer = STFT(
            frame_length=config.window_length,
            fft_length=config.n_fft,
            frame_step=config.hop_length,
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
        real = tf.math.real(stft)
        real = tf.where(tf.math.is_nan(real) | tf.math.is_inf(real), tf.zeros_like(real), real)

        imag = tf.math.imag(stft)
        imag = tf.where(tf.math.is_nan(imag) | tf.math.is_inf(imag), tf.zeros_like(imag), imag)
        return tf.concat([real, imag], axis=-1)

    def call(self, inputs, training=False):
        inputs = self.preprocess(inputs)
        # inputs: (B, T, Fr, C)
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        inputs = (inputs - mean) / (std + 1e-5)
        inputs, skip_list, lengths_list, original_lengths_list = self.encoder(inputs, training=training)
        return inputs, skip_list, lengths_list, original_lengths_list, (mean, std)


class AcousticModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_separate = 2

        self.encoder = AcousticEncoderModel(config)
        self.decoder = Decoder(
            hidden_size_list=config.hidden_size_list,
            band_SR=config.band_SR,
            band_strides=config.band_strides,
            band_kernels=config.band_kernels,
            num_separate=self.num_separate,
        )

        self.num_channels = config.num_channels
        self.istft_n_fft = config.n_fft
        self.istft_hop_length = config.hop_length
        self.istft_window_length = config.window_length
        self.sampling_rate = config.sampling_rate

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
        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        F = tf.shape(inputs)[2]
        C = tf.shape(inputs)[3]
        # (B, T, F, C) -> (B, T, F, C // num_separate, num_separate)
        inputs = tf.reshape(
            inputs,
            [B, T, F, C // self.num_separate, self.num_separate],
        )
        inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
        inputs = tf.reshape(
            inputs,
            [B * self.num_separate, T, F, C // self.num_separate],
        )

        if use_padding:
            crop_length = tf.cast(tf.math.ceil(seq_len / self.istft_hop_length), dtype=tf.int32)
            padding_num = tf.maximum(tf.shape(inputs)[1] - crop_length, 0)
            inputs = tf.pad(inputs, [[0, 0], [0, padding_num], [0, 0], [0, 0]], mode="REFLECT")
            inputs = inputs[:, :crop_length]

        # vocoderで使われる位相・振幅予測では安定性が低いため、complexを直接予測する
        split_inputs = tf.split(inputs, num_or_size_splits=2 * self.num_channels, axis=-1)
        real = tf.concat(split_inputs[: self.num_channels], axis=-1)
        imag = tf.concat(split_inputs[self.num_channels :], axis=-1)

        inputs = inverse_stft_complex(
            tf.complex(real, imag),
            frame_length=self.istft_window_length,
            frame_step=self.istft_hop_length,
            fft_length=self.istft_n_fft,
        )
        inputs = tf.reshape(inputs, [B, self.num_separate, tf.shape(inputs)[1], tf.shape(inputs)[2]])
        inputs = inputs[:, :, :seq_len]
        return inputs

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]

        inputs, skip_list, lengths_list, original_lengths_list, (mean, std) = self.encoder(inputs, training=training)
        # inputs: (batch, time, freq, channels)
        inputs = self.decoder(
            inputs,
            skip_list=skip_list,
            lengths_list=lengths_list,
            original_lengths_list=original_lengths_list,
            training=training,
        )
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = inputs * std + mean
        inputs = self.inverse(inputs, use_padding=True, seq_len=seq_len)
        return inputs
