import tensorflow as tf

from librosa.filters import mel as mel_filter


def log10(inputs):
    return tf.math.log(inputs) / tf.math.log(tf.constant(10, dtype=inputs.dtype))


def dbscale(inputs, top_db=80):
    inputs = tf.math.square(inputs)
    inputs = 10.0 * log10(inputs)
    inputs = tf.math.maximum(inputs, tf.math.reduce_max(inputs) - top_db)
    return inputs


def inverse_stft(x, phase, frame_length, frame_step, fft_length):
    x = tf.signal.inverse_stft(
        tf.transpose(tf.complex(x * tf.cos(phase), x * tf.sin(phase)), [3, 0, 1, 2]),
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.inverse_stft_window_fn(frame_step),
    )
    x = tf.transpose(x, [1, 2, 0])
    return x


class STFT(tf.keras.layers.Layer):
    def __init__(
        self, frame_length, fft_length=None, frame_step=None, logscale=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.frame_step = frame_step
        self.logscale = logscale

        if self.fft_length is None:
            self.fft_length = self.frame_length
        if self.frame_step is None:
            self.frame_step = self.frame_length // 4

    def call(self, signals, use_padding=True, return_complex=False):
        batch_size = tf.shape(signals)[0]
        num_channels = tf.shape(signals)[-1]

        signals = tf.transpose(signals, [0, 2, 1])
        signals = tf.reshape(signals, [-1, tf.shape(signals)[-1]])

        if use_padding:
            padding_num = round(self.fft_length - self.frame_step)
            signals = tf.pad(signals, [[0, 0], [padding_num, 0]], mode="REFLECT")
        stft = tf.signal.stft(
            signals=signals,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            pad_end=use_padding,
        )
        stft = tf.reshape(
            stft, [batch_size, num_channels, tf.shape(stft)[-2], tf.shape(stft)[-1]]
        )
        stft = tf.transpose(stft, [0, 2, 3, 1])

        if return_complex:
            return stft

        mag = tf.math.abs(stft)
        if self.logscale:
            mag = tf.math.log(tf.clip_by_value(mag, 1e-5, 1e3))
        phase = tf.math.angle(stft)
        return mag, phase


class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(
        self,
        n_mels,
        window_length,
        sampling_rate,
        fmin=0,
        fmax=8000,
        n_fft=None,
        hop_length=None,
        logscale=True,
        dbscale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.window_length = window_length
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.logscale = logscale
        self.dbscale = dbscale

        if self.n_fft is None:
            self.n_fft = window_length
        if self.hop_length is None:
            self.hop_length = window_length // 4

        self.lin_to_mel_matrix = tf.convert_to_tensor(
            mel_filter(
                sr=sampling_rate,
                n_fft=self.n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                htk=False,
                norm="slaney",
            ).T,
            dtype=tf.float32,
        )

    def call(self, signals, use_padding=True):
        batch_size = tf.shape(signals)[0]
        num_channels = tf.shape(signals)[-1]

        signals = tf.transpose(signals, [0, 2, 1])
        signals = tf.reshape(signals, [-1, tf.shape(signals)[-1]])

        if use_padding:
            padding_num = round(self.n_fft - self.hop_length)
            signals = tf.pad(signals, [[0, 0], [padding_num, 0]], mode="REFLECT")
        stft = tf.signal.stft(
            signals=signals,
            frame_length=self.window_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            pad_end=True,
        )
        stft = tf.math.abs(stft)
        mel_spectrogram = tf.tensordot(stft, self.lin_to_mel_matrix, axes=1)
        if self.dbscale:
            mel_spectrogram = dbscale(mel_spectrogram)
        elif self.logscale:
            mel_spectrogram = tf.math.log(
                tf.clip_by_value(mel_spectrogram, 1e-9, mel_spectrogram)
            )
        mel_spectrogram = tf.reshape(
            mel_spectrogram,
            [
                batch_size,
                num_channels,
                tf.shape(mel_spectrogram)[-2],
                tf.shape(mel_spectrogram)[-1],
            ],
        )
        mel_spectrogram = tf.transpose(mel_spectrogram, [0, 2, 3, 1])
        mel_spectrogram = tf.where(
            tf.math.is_nan(mel_spectrogram) | tf.math.is_inf(mel_spectrogram),
            tf.zeros_like(mel_spectrogram),
            mel_spectrogram,
        )
        return mel_spectrogram


class MelSpectrogramLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        n_mels_list,
        window_length_list,
        sampling_rate,
        fmin=[
            0,
        ],
        fmax=[
            None,
        ],
        loss_weights=[
            1,
        ],
        logscale=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate

        self.loss_weights = loss_weights
        self.mel_layers = [
            MelSpectrogram(
                n_mels=n_mels,
                window_length=window_length,
                sampling_rate=sampling_rate,
                fmin=fmin,
                fmax=fmax,
                logscale=logscale,
                dtype=tf.float32,
            )
            for n_mels, window_length, fmin, fmax in zip(
                n_mels_list, window_length_list, fmin, fmax
            )
        ]

    def call(self, y, x):
        total_loss = 0

        mel_inputs = []
        mel_preds = []
        for i, mel_layer in enumerate(self.mel_layers):
            x_mels = mel_layer(x)
            y_mels = mel_layer(y)

            mel_preds.append(x_mels)
            mel_inputs.append(y_mels)
            loss = tf.reduce_mean(tf.abs(y_mels - x_mels)) * self.loss_weights[i]
            total_loss += loss

        return total_loss, mel_inputs, mel_preds
