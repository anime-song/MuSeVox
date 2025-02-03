import cv2
import os
import argparse

# 不要なログ対策
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from omegaconf import OmegaConf

from acoustic_token_models.dataset import DataGeneratorBatch, load_from_npz
from util import allocate_gpu_memory, minmax
from losses import MelSpectrogramLoss
from acoustic_token_models.acoustic_token_model import AcousticModel
from acoustic_token_models.discriminator import Discriminator


class TensorboardCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_dir,
        test_data,
        base_model,
        sampling_rate=22050,
        period=1,
        sample_period=1,
        initial_step=0,
        mel_func=None,
    ):
        self.summary_writer = tf.summary.create_file_writer(log_dir, max_queue=100)
        self.period = period
        self.sample_period = sample_period
        self.current_steps = initial_step
        self.current_test_steps = initial_step

        self.base_model = base_model
        self.x_data, self.y_data = test_data.__getitem__(0)
        self.sampling_rate = sampling_rate
        self.first_call = True
        self.mel_func = mel_func

    def on_train_batch_end(self, batch, logs=None):
        if logs:
            for key, value in logs.items():
                with self.summary_writer.as_default():
                    tf.summary.scalar(key, value, step=self.current_steps)

        if self.current_steps % self.sample_period == 0:
            piano = self.x_data[0]
            mix = self.x_data[1]
            piano_mix = piano + mix

            separated = self.base_model(piano_mix, training=False)
            separated_piano = separated[:, 0]
            separated_other = separated[:, 1]

            _, piano_mel_inputs, piano_mel_preds = self.mel_func(piano, separated_piano)
            _, mix_mel_inputs, mix_mel_preds = self.mel_func(mix, separated_other)

            if self.first_call:
                cv2.imwrite(
                    "./img/orig_piano.png",
                    cv2.flip(
                        minmax(piano_mel_inputs[-1].numpy()[0, :, :, 0].transpose(1, 0)),
                        0,
                    )
                    * 255,
                )
                cv2.imwrite(
                    "./img/orig_other.png",
                    cv2.flip(
                        minmax(mix_mel_inputs[-1].numpy()[0, :, :, 0].transpose(1, 0)),
                        0,
                    )
                    * 255,
                )
                self.first_call = False

            cv2.imwrite(
                "./img/separated_piano_{}.png".format(self.current_steps),
                cv2.flip(minmax(piano_mel_preds[-1].numpy()[0, :, :, 0].transpose(1, 0)), 0) * 255,
            )
            cv2.imwrite(
                "./img/separated_mix_{}.png".format(self.current_steps),
                cv2.flip(minmax(mix_mel_preds[-1].numpy()[0, :, :, 0].transpose(1, 0)), 0) * 255,
            )

        self.current_steps += 1

    def on_test_batch_end(self, batch, logs=None):
        if logs:
            for key, value in logs.items():
                with self.summary_writer.as_default():
                    tf.summary.scalar(key, value, step=self.current_test_steps)

        self.current_test_steps += 1


class AcousticTokenTrainer:
    def __init__(
        self,
        epochs: int,
        generator_model: tf.keras.Model,
        discriminator_model: tf.keras.Model,
        generator_optimizer: tf.keras.optimizers.Optimizer,
        discriminator_optimizer: tf.keras.optimizers.Optimizer,
        callbacks: list,
        discriminator_callbacks: list,
        mel_loss_weight: float = 1.0,
        feature_loss_weight: float = 3.0,
        generator_loss_weight: float = 3.0,
        mrd_loss_weight: float = 0.1,
        initial_step: int = 0,
        warmup: bool = False,
        discriminator_update_prob: float = 0.66,
        initial_epoch: int = 0,
        discriminator_loss_type: str = "mse",
        loss_func=None,
    ):
        self.generator = generator_model
        self.discriminator = discriminator_model

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=False,
            model=self.generator,
        )
        self.discriminator_callbacks = tf.keras.callbacks.CallbackList(
            callbacks=discriminator_callbacks,
            add_history=True,
            add_progbar=False,
            model=self.discriminator,
        )
        self.epochs = epochs
        self.mel_loss_weight = mel_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.generator_loss_weight = generator_loss_weight
        self.mrd_loss_weight = mrd_loss_weight
        self.current_step = initial_step
        self.initial_epoch = initial_epoch
        self.warmup = warmup
        self.discriminator_update_prob = discriminator_update_prob
        self.discriminator_loss_type = discriminator_loss_type
        self.loss_func = loss_func

    def discriminator_loss(self, fake_preds, real_preds):
        total_fake_loss = 0.0
        total_real_loss = 0.0
        for fake_pred, real_pred in zip(fake_preds, real_preds):
            if self.discriminator_loss_type == "mse":
                total_real_loss += tf.reduce_mean((1 - real_pred) ** 2)
                total_fake_loss += tf.reduce_mean(fake_pred**2)
            elif self.discriminator_loss_type == "hinge":
                total_real_loss += tf.reduce_mean(tf.nn.relu(1 - real_pred))
                total_fake_loss += tf.reduce_mean(tf.nn.relu(1 + fake_pred))
        return total_real_loss, total_fake_loss

    def generator_loss(self, fake_preds):
        total_loss = 0.0
        for fake_pred in fake_preds:
            if self.discriminator_loss_type == "mse":
                total_loss += tf.reduce_mean((1 - fake_pred) ** 2)
            elif self.discriminator_loss_type == "hinge":
                total_loss += tf.reduce_mean(tf.nn.relu(1 - fake_pred))
        return total_loss

    def feature_matching_loss(self, fake_intermediates, real_intermediates):
        losses = [
            tf.reduce_mean(tf.abs(tf.cast(real, tf.float32) - tf.cast(fake, tf.float32)))
            for fake, real in zip(fake_intermediates, real_intermediates)
        ]
        return tf.reduce_sum(losses)

    def mel_loss(self, original_inputs, inputs):
        mel_loss, _, _ = self.loss_func(original_inputs, inputs)
        return mel_loss

    def calculate_sdr(self, s_true, s_estimated):
        def log10(inputs):
            return tf.math.log(inputs) / tf.math.log(tf.constant(10, dtype=inputs.dtype))

        """
        SDR（信号対歪み比）を計算する関数
        s_true: 正解の音源
        s_estimated: 推定された音源
        """
        s_error = s_estimated - s_true
        s_true_power = tf.reduce_sum(s_true**2, axis=[1, 2])
        s_error_power = tf.reduce_sum(s_error**2, axis=[1, 2])

        # SDR 計算（対数変換）
        sdr = 10 * log10(s_true_power / (s_error_power + 1e-10))  # 0 除算を防ぐ
        return tf.reduce_mean(sdr)

    def train(self, train_gen, test_gen):
        logs = {}
        self.callbacks.on_train_begin(logs)
        self.discriminator_callbacks.on_train_begin(logs)

        for epoch in range(self.initial_epoch, self.epochs):
            self.callbacks.on_epoch_begin(epoch, logs=logs)
            self.discriminator_callbacks.on_epoch_begin(epoch, logs=logs)

            # training step
            for i, input_data in enumerate(train_gen):
                self.callbacks.on_train_batch_begin(i, logs=logs)
                self.discriminator_callbacks.on_train_batch_begin(epoch, logs=logs)

                logs = self.train_step(input_data)

                self.callbacks.on_train_batch_end(i, logs=logs)
                self.discriminator_callbacks.on_train_batch_end(i, logs=logs)
                self.current_step += 1

            # test step
            for i, input_data in enumerate(test_gen):
                self.callbacks.on_test_batch_begin(i, logs=logs)
                self.discriminator_callbacks.on_test_batch_begin(i, logs=logs)

                logs = self.test_step(input_data)

                self.callbacks.on_test_batch_end(i, logs=logs)
                self.discriminator_callbacks.on_test_batch_end(i, logs=logs)
            train_gen.on_epoch_end()
            test_gen.on_epoch_end()
            self.callbacks.on_epoch_end(epoch, logs=logs)
            self.discriminator_callbacks.on_epoch_end(epoch, logs=logs)

        self.callbacks.on_train_end(logs)
        self.discriminator_callbacks.on_train_end(logs)

    @tf.function
    def train_step(self, input_data):
        x_batch_train, _ = input_data
        x_train_piano = x_batch_train[0]
        x_train_mix = x_batch_train[1]

        x_train_piano_mix = x_train_piano + x_train_mix

        # Discriminatorのトレーニング
        with (
            tf.GradientTape() as gen_tape,
            tf.GradientTape() as disc_tape,
        ):
            separated = self.generator(x_train_piano_mix, training=True)
            separated_piano = separated[:, 0]
            separated_other = separated[:, 1]
            g_fake = tf.concat([separated_piano, separated_other], axis=-1)
            g_real = tf.concat([x_train_piano, x_train_mix], axis=-1)
            fake_pred, fake_intermediates = self.discriminator(g_fake, training=True)
            real_pred, real_intermediates = self.discriminator(g_real, training=True)

            mpd_d_real_loss, mpd_d_fake_loss = self.discriminator_loss(fake_pred["mpd"], real_pred["mpd"])
            mrd_d_real_loss, mrd_d_fake_loss = self.discriminator_loss(fake_pred["mrd"], real_pred["mrd"])
            d_scaled_loss = self.discriminator_optimizer.get_scaled_loss(
                mpd_d_real_loss
                + mrd_d_real_loss * self.mrd_loss_weight
                + mpd_d_fake_loss
                + mrd_d_fake_loss * self.mrd_loss_weight
            )

            mpd_feature_loss = self.feature_matching_loss(fake_intermediates["mpd"], real_intermediates["mpd"])
            mrd_feature_loss = self.feature_matching_loss(fake_intermediates["mrd"], real_intermediates["mrd"])
            feature_loss = mpd_feature_loss + mrd_feature_loss * self.mrd_loss_weight

            g_loss = (
                self.generator_loss(fake_pred["mpd"]) + self.generator_loss(fake_pred["mrd"]) * self.mrd_loss_weight
            )
            mel_loss = self.mel_loss(x_train_piano, separated_piano) + self.mel_loss(x_train_mix, separated_other)
            scaling_loss = tf.reduce_sum(self.generator.losses)
            piano_sdr = self.calculate_sdr(x_train_piano, separated_piano)
            other_sdr = self.calculate_sdr(x_train_mix, separated_other)

            total_generator_loss = (
                mel_loss * self.mel_loss_weight
                + g_loss * self.generator_loss_weight
                + feature_loss * self.feature_loss_weight
                + scaling_loss
            )
            total_generator_loss = self.generator_optimizer.get_scaled_loss(total_generator_loss)

        # Discriminatorが強すぎることを考慮して一定の確率で重みを更新する
        random_value = tf.random.uniform([], minval=0.0, maxval=1.0)
        if random_value < self.discriminator_update_prob:
            disc_grads = disc_tape.gradient(d_scaled_loss, self.discriminator.trainable_variables)
            disc_grads = self.discriminator_optimizer.get_unscaled_gradients(disc_grads)
            self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        gen_grads = gen_tape.gradient(total_generator_loss, self.generator.trainable_variables)
        gen_grads = self.generator_optimizer.get_unscaled_gradients(gen_grads)
        self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        metrics = {
            "d_real_loss_mpd": mpd_d_real_loss / len(real_pred["mpd"]),
            "d_fake_loss_mpd": mpd_d_fake_loss / len(fake_pred["mpd"]),
            "d_real_loss_mrd": mrd_d_real_loss / len(real_pred["mrd"]),
            "d_fake_loss_mrd": mrd_d_fake_loss / len(fake_pred["mrd"]),
            "g_loss": g_loss,
            "feature_loss_mpd": mpd_feature_loss,
            "feature_loss_mrd": mrd_feature_loss,
            "mel_loss": mel_loss,
            "g_scaling_loss": scaling_loss,
            "piano_sdr": piano_sdr,
            "other_sdr": other_sdr,
        }
        return metrics

    @tf.function
    def test_step(self, input_data):
        x_batch_test, _ = input_data
        x_test_piano = x_batch_test[0]
        x_test_mix = x_batch_test[1]

        x_test_piano_mix = x_test_piano + x_test_mix
        separated = self.generator(x_test_piano_mix, training=False)
        separated_piano = separated[:, 0]
        separated_other = separated[:, 1]

        mel_loss = self.mel_loss(x_test_piano, separated_piano) + self.mel_loss(x_test_mix, separated_other)
        piano_sdr = self.calculate_sdr(x_test_piano, separated_piano)
        other_sdr = self.calculate_sdr(x_test_mix, separated_other)

        return {"val_mel_loss": mel_loss, "val_piano_sdr": piano_sdr, "val_other_sdr": other_sdr}


def get_dataset(
    batch_size,
    patch_len,
    cache_size,
    epoch_max_steps,
    sampling_rate,
    mix_folder_path,
    piano_folder_path,
    instruments_folder_path,
):
    """
    Load and prepare the dataset for training and testing.

    Args:
        batch_size (int): The size of the batches.
        patch_len (int): The length of the patches.
        cache_size (int): The size of the cache.
        epoch_max_steps (int): The maximum number of steps per epoch.
        sampling_rate (int): The sampling rate of the audio data.

    Returns:
        tuple: A tuple containing the training, testing, and plotting data generators.
    """
    x_train_piano, x_test_piano, dataset_piano = load_from_npz(directory=piano_folder_path, group_name="piano")
    x_train_mix, x_test_mix, dataset_mix = load_from_npz(directory=mix_folder_path, group_name="mix")
    x_train_instruments, x_test_instruments, dataset_instruments = load_from_npz(
        directory=instruments_folder_path, group_name="instruments"
    )

    train_gen = DataGeneratorBatch(
        piano_files=x_train_piano,
        mix_files=x_train_mix,
        instruments_files=x_train_instruments,
        dataset_piano=dataset_piano,
        dataset_mix=dataset_mix,
        dataset_instruments=dataset_instruments,
        sampling_rate=sampling_rate,
        patch_length=patch_len,
        initial_epoch=0,
        max_queue=2,
        cache_size=cache_size,
        batch_size=batch_size,
        epoch_max_steps=epoch_max_steps,
    )

    test_gen = DataGeneratorBatch(
        piano_files=x_test_piano,
        mix_files=x_test_mix,
        instruments_files=x_test_instruments,
        dataset_piano=dataset_piano,
        dataset_mix=dataset_mix,
        dataset_instruments=dataset_instruments,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        patch_length=patch_len,
        cache_size=cache_size,
        epoch_max_steps=1000,
    )

    plot_gen = DataGeneratorBatch(
        piano_files=x_test_piano,
        mix_files=x_test_mix,
        instruments_files=x_test_instruments,
        dataset_piano=dataset_piano,
        dataset_mix=dataset_mix,
        dataset_instruments=dataset_instruments,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        patch_length=patch_len,
        cache_size=cache_size,
    )

    return train_gen, test_gen, plot_gen


def train(
    config,
    config_name,
    model_weight_path=None,
    discriminator_model_weight_path=None,
):
    # モデル構築
    gen_model, gen_intermediate_model = AcousticModel.from_pretrain(
        config=config,
        model_weight_path=model_weight_path,
    )
    gen_model.summary()

    disc_model, _ = Discriminator.from_pretrain(config=config, model_weight_path=discriminator_model_weight_path)

    # Callback
    monitor = "val_mel_loss"
    ckpt_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/musevox_model-epoch_{epoch}_step_{batch}/generator.ckpt",
        monitor=monitor,
        verbose=1,
        save_weights_only=True,
        save_freq=config.model_save_freq,
    )

    discriminator_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./model/musevox_model-epoch_{epoch}_step_{batch}/discriminator.ckpt",
            monitor=monitor,
            verbose=1,
            save_weights_only=True,
            save_freq=config.model_save_freq,
        )
    ]

    # Optimzier
    g_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-4, decay_steps=1, decay_rate=0.999999, staircase=False
    )
    g_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=g_lr_schedule,
        beta_1=0.9,
        beta_2=0.95,
        global_clipnorm=100,
        weight_decay=0.04,
    )
    g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(g_optimizer)

    initial_step = config.initial_step
    # 通常の訓練
    train_gen, test_gen, plot_gen = get_dataset(
        config.batch_size,
        config.patch_len,
        config.cache_size,
        config.epoch_max_steps,
        config.sampling_rate,
        config.mix_folder_path,
        config.piano_folder_path,
        config.instruments_folder_path,
    )
    mel_loss_func = MelSpectrogramLoss(
        config.mel_loss_n_mels,
        config.mel_loss_window_length,
        config.sampling_rate,
        fmin=config.mel_loss_fmin,
        fmax=config.mel_loss_fmax,
        loss_weights=config.mel_loss_weights,
        dtype=tf.float32,
    )

    progbar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
    progbar.set_params({"verbose": 1, "epochs": config.epochs, "steps": len(train_gen)})
    tensorboard_callback = TensorboardCallback(
        log_dir=os.path.join(config.log_dir, config_name),
        test_data=plot_gen,
        base_model=gen_intermediate_model,
        sampling_rate=config.sampling_rate,
        period=10,
        sample_period=config.plot_interval_step,
        initial_step=initial_step,
        mel_func=mel_loss_func,
    )
    d_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-4, decay_steps=1, decay_rate=0.999999, staircase=False
    )
    d_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=d_lr_schedule,
        beta_1=0.9,
        beta_2=0.95,
        global_clipnorm=100,
        weight_decay=0.04,
    )
    d_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(d_optimizer)

    print(f"Discriminatorの更新確率: {config.discriminator_update_prob}")
    AcousticTokenTrainer(
        epochs=config.epochs,
        generator_model=gen_model,
        discriminator_model=disc_model,
        generator_optimizer=g_optimizer,
        discriminator_optimizer=d_optimizer,
        callbacks=[
            ckpt_callback_best,
            progbar,
            tensorboard_callback,
        ],
        discriminator_callbacks=discriminator_callbacks,
        mel_loss_weight=config.mel_loss_weight,
        feature_loss_weight=config.feature_loss_weight,
        generator_loss_weight=config.generator_loss_weight,
        mrd_loss_weight=config.mrd_loss_weight,
        initial_step=initial_step,
        warmup=False,
        discriminator_update_prob=config.discriminator_update_prob,
        initial_epoch=initial_step // config.epoch_max_steps,
        discriminator_loss_type=config.discriminator_loss_type,
        loss_func=mel_loss_func,
    ).train(train_gen, test_gen)


if __name__ == "__main__":
    allocate_gpu_memory()

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)

    os.makedirs("./model", exist_ok=True)
    os.makedirs("./img", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="コンフィグファイルのファイルパス",
    )
    parser.add_argument(
        "-g_p",
        "--g_checkpoint_path",
        type=str,
        default=None,
        help="ジェネレータのモデルの重みのパス",
    )
    parser.add_argument(
        "-d_p",
        "--d_checkpoint_path",
        type=str,
        default=None,
        help="ディスクリミネーターのモデルの重みのパス",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    train(
        config=config,
        config_name=config_name,
        model_weight_path=args.g_checkpoint_path,
        discriminator_model_weight_path=args.d_checkpoint_path,
    )
