import einops
import tensorflow as tf

from model_utils import WeightNormalization


def get_activation_layer(activation):
    if activation == "leakyrelu":
        return tf.keras.layers.LeakyReLU(0.1)
    else:
        return tf.keras.layers.Activation(activation)


class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, kernel_size, strides, padding="same", activation="leakyrelu", **kwargs):
        super().__init__(**kwargs)
        self.conv = WeightNormalization(
            tf.keras.layers.Conv2D(hidden_size, kernel_size, strides, padding=padding), data_init=False
        )
        self.activation = get_activation_layer(activation)

    def call(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.activation(inputs)
        return inputs


class MRD(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        fft_size,
        hop_length,
        window_len,
        num_channels=2,
        generator_hop_length=256,
        activation="leakyrelu",
        scaling_loss_weight=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fft_size = fft_size
        self.hop_length = hop_length
        self.window_len = window_len
        self.num_channels = num_channels
        self.generator_hop_length = generator_hop_length

        self.layers = [
            Conv2DBlock(channels, (9, 3), (1, 1), padding="same", activation=activation),
            Conv2DBlock(channels, (9, 3), (2, 1), padding="same", activation=activation),
            Conv2DBlock(channels, (9, 3), (2, 1), padding="same", activation=activation),
            Conv2DBlock(channels, (9, 3), (2, 1), padding="same", activation=activation),
            Conv2DBlock(channels, (3, 3), (1, 1), padding="same", activation=activation),
        ]

        self.last_conv = WeightNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding="same", dtype=tf.float32),
            data_init=False,
            dtype=tf.float32,
        )

    def get_spectrogram(self, signals):
        signals = tf.cast(signals, dtype=tf.float32)
        batch_size = tf.shape(signals)[0]
        num_channels = tf.shape(signals)[-1]

        signals = tf.transpose(signals, [0, 2, 1])
        signals = tf.reshape(signals, [-1, tf.shape(signals)[-1]])

        stft = tf.signal.stft(
            signals=signals, frame_length=self.window_len, frame_step=self.hop_length, fft_length=self.fft_size
        )
        stft = tf.reshape(stft, [batch_size, num_channels, tf.shape(stft)[-2], tf.shape(stft)[-1]])
        stft = tf.transpose(stft, [0, 2, 3, 1])
        stft = tf.abs(stft)
        stft = tf.where(tf.math.is_nan(stft) | tf.math.is_inf(stft), tf.zeros_like(stft), stft)
        return stft

    def call(self, inputs):
        inputs = self.get_spectrogram(inputs)
        intermediate_layers = []

        for layer in self.layers:
            inputs = layer(inputs)
            intermediate_layers.append(inputs)

        inputs = self.last_conv(inputs)
        return [
            inputs,
        ], intermediate_layers


class MBMRD(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        fft_size,
        hop_length,
        window_len,
        num_channels=2,
        generator_hop_length=256,
        activation="leakyrelu",
        scaling_loss_weight=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fft_size = fft_size
        self.hop_length = hop_length
        self.window_len = window_len
        self.num_channels = num_channels
        self.generator_hop_length = generator_hop_length

        n_bins = fft_size // 2 + 1
        bands_ratio = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        self.bands = [(int(b[0] * n_bins), int(b[1] * n_bins)) for b in bands_ratio]

        self.band_convs = []
        for _ in range(len(self.bands)):
            # UnivNetのものと次元が逆だがDACに合わせる
            self.band_convs.append(
                [
                    Conv2DBlock(channels, (3, 9), (1, 1), padding="same", activation=activation),
                    Conv2DBlock(channels, (3, 9), (1, 2), padding="same", activation=activation),
                    Conv2DBlock(channels, (3, 9), (1, 2), padding="same", activation=activation),
                    Conv2DBlock(channels, (3, 9), (1, 2), padding="same", activation=activation),
                    Conv2DBlock(channels, (3, 3), (1, 1), padding="same", activation=activation),
                ]
            )

        self.last_conv = WeightNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding="same", dtype=tf.float32),
            data_init=False,
            dtype=tf.float32,
        )

    def get_spectrogram(self, signals):
        signals = tf.cast(signals, dtype=tf.float32)
        batch_size = tf.shape(signals)[0]
        num_channels = tf.shape(signals)[-1]

        signals = tf.transpose(signals, [0, 2, 1])
        signals = tf.reshape(signals, [-1, tf.shape(signals)[-1]])

        padding_num = round(self.fft_size - self.hop_length)
        signals = tf.pad(signals, [[0, 0], [padding_num, 0]], mode="REFLECT")
        stft = tf.signal.stft(
            signals=signals, frame_length=self.window_len, frame_step=self.hop_length, fft_length=self.fft_size
        )
        stft = tf.reshape(stft, [batch_size, num_channels, tf.shape(stft)[-2], tf.shape(stft)[-1]])
        stft = tf.transpose(stft, [0, 2, 3, 1])
        stft = tf.concat([tf.math.real(stft), tf.math.imag(stft)], axis=-1)
        stft = tf.where(tf.math.is_nan(stft) | tf.math.is_inf(stft), tf.zeros_like(stft), stft)
        stft_bands = [stft[..., b[0] : b[1], :] for b in self.bands]
        return stft_bands

    def call(self, inputs):
        inputs_bands = self.get_spectrogram(inputs)
        intermediate_layers = []

        out_stack = []
        for band, conv_layers in zip(inputs_bands, self.band_convs):
            for layer in conv_layers:
                band = layer(band)
                intermediate_layers.append(band)
            out_stack.append(band)

        output = tf.concat(out_stack, axis=-2)
        output = self.last_conv(output)
        return [
            output,
        ], intermediate_layers


class MPD(tf.keras.layers.Layer):
    def __init__(
        self, period, initial_channels=4, num_channels=2, generator_hop_length=256, activation="leakyrelu", **kwargs
    ):
        super().__init__(**kwargs)
        self.period = period

        self.layers = [
            Conv2DBlock(initial_channels, (5, 1), (3, 1), padding="same", activation=activation),
            Conv2DBlock(initial_channels * 4, (5, 1), (3, 1), padding="same", activation=activation),
            Conv2DBlock(initial_channels * 16, (5, 1), (3, 1), padding="same", activation=activation),
            Conv2DBlock(initial_channels * 32, (5, 1), (3, 1), padding="same", activation=activation),
            Conv2DBlock(initial_channels * 32, (5, 1), (1, 1), padding="same", activation=activation),
        ]

        self.last_conv = WeightNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=(3, 1), padding="same", dtype=tf.float32),
            data_init=False,
            dtype=tf.float32,
        )

        self.num_channels = num_channels
        self.generator_hop_length = generator_hop_length

    def call(self, inputs):
        intermediate_layers = []

        padding_num = self.period - (tf.shape(inputs)[1] % self.period)
        inputs = tf.pad(inputs, [[0, 0], [0, padding_num], [0, 0]], mode="REFLECT")
        inputs = einops.rearrange(inputs, "b (t p) c -> b t p c", p=self.period)

        for layer in self.layers:
            inputs = layer(inputs)
            intermediate_layers.append(inputs)

        inputs = self.last_conv(inputs)
        inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        return inputs, intermediate_layers


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.mpd_layers = [
            MPD(
                period,
                initial_channels=32,
                num_channels=config.num_channels,
                generator_hop_length=config.hop_length,
                activation=config.discriminator_activation,
            )
            for period in config.discriminator_periods
        ]
        self.mrd_layers = [
            MBMRD(
                32,
                fft_size=fft_size,
                hop_length=hop_length,
                window_len=win_len,
                num_channels=config.num_channels,
                generator_hop_length=config.hop_length,
                activation=config.discriminator_activation,
            )
            for fft_size, hop_length, win_len in config.discriminator_stft_params
        ]

    @classmethod
    def from_pretrain(cls, config, model_weight_path=None):
        model_input = tf.keras.layers.Input(shape=(None, config.num_channels * 2))

        intermediate_model = cls(config)
        outputs = intermediate_model(model_input)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)

        if model_weight_path is not None:
            model.load_weights(model_weight_path)

        return model, intermediate_model

    def preprocess(self, y):
        # Remove DC offset
        y = y - tf.reduce_mean(y, axis=1, keepdims=True)

        # Peak normalize the volume of input audio
        max_val = tf.reduce_max(tf.abs(y), axis=1, keepdims=True)
        y = 0.8 * y / (max_val + 1e-9)
        return y

    def call(self, inputs, training=False):
        inputs = self.preprocess(inputs)
        mpd_outputs = []
        mpd_intermediate_layers = []
        for discriminator in self.mpd_layers:
            output, intermediate_layer = discriminator(inputs, training=training)
            mpd_intermediate_layers.extend(intermediate_layer)
            mpd_outputs.append(output)

        mrd_output = []
        mrd_intermediate_layers = []
        for discriminator in self.mrd_layers:
            output, intermediate_layer = discriminator(inputs, training=training)
            mrd_intermediate_layers.extend(intermediate_layer)
            mrd_output.extend(output)

        return {"mpd": mpd_outputs, "mrd": mrd_output}, {"mpd": mpd_intermediate_layers, "mrd": mrd_intermediate_layers}
