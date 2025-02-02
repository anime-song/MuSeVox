import tensorflow as tf
import numpy as np
import pyloudnorm as pyln


def preprocess_wav(y, sampling_rate, target_loudness=-24.0):
    meter = pyln.Meter(sampling_rate)
    loudness = meter.integrated_loudness(y.T)
    y = pyln.normalize.loudness(y.T, loudness, target_loudness)

    gain_db = target_loudness - loudness
    gain_factor = 10 ** (gain_db / 20)
    return y.T, gain_db, gain_factor


def postprocess_wav(y, sampling_rate, gain_db):
    inverse_gain_factor = 10 ** (-gain_db / 20)
    return np.clip(y * inverse_gain_factor, -1.0, 1.0)


def minmax(x, axis=None, min=None, max=None):
    x_min = min
    x_max = max

    if x_min is None:
        x_min = x.min(axis=axis, keepdims=True)

    if x_max is None:
        x_max = x.max(axis=axis, keepdims=True)

    return (x - x_min) / (x_max - x_min)


def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, start_lr=0.0, target_lr=1e-3):
    learning_rate = (
        0.5
        * target_lr
        * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    )
    warmup_lr = target_lr * (global_step / warmup_steps)
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold, learning_rate, target_lr)
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(
        self,
        total_steps=0,
        warmup_steps=0,
        start_lr=0.0,
        target_lr=1e-3,
        hold=0,
        global_steps=0,
    ):
        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = global_steps
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(
            global_step=self.global_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class MultiStepLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        base_learning_rate,
        step_list,
        gamma=0.5,
        initial_step=0,
        warmup_steps=0,
        name=None,
    ):
        super().__init__()
        self.base_learning_rate = tf.convert_to_tensor(base_learning_rate, dtype=tf.float32)
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        self.name = name
        self.step_list = tf.convert_to_tensor(step_list, dtype=tf.int32)
        self.warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.int32)
        self.current_step = tf.Variable(initial_value=initial_step, trainable=False, dtype=tf.int32)

    def add_step(self):
        self.current_step.assign_add(1)

    def __call__(self, step):
        with tf.name_scope(self.name or "MultiStepLR") as name:
            step = tf.cast(self.current_step, tf.int32)
            current_interval = tf.reduce_sum(tf.cast(step >= self.step_list, tf.int32))
            learning_rate = self.base_learning_rate * tf.pow(self.gamma, tf.cast(current_interval, tf.float32))

            return tf.cond(
                step < self.warmup_steps,
                lambda: self.base_learning_rate * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)),
                lambda: learning_rate,
                name=name,
            )


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")

    if len(physical_devices) > 0:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")
