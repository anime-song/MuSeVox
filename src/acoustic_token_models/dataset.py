import time
import random
import threading
import copy
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import h5py


def load_from_npz(directory="./Dataset/Processed", group_name="musics"):
    h5 = h5py.File(directory + "/dataset.hdf5", mode="r")
    group = h5.require_group(group_name)

    # メモリーリークの可能性あり
    datasetList = [name for name in group if isinstance(group[name], h5py.Dataset)]

    dataset = group

    train_files, test_files = train_test_split(datasetList, test_size=0.01, random_state=42)

    return train_files, test_files, dataset


def load_data(self):
    while not self.is_epoch_end:
        should_added_queue = len(self.data_cache_queue) < self.max_queue
        while should_added_queue:
            self._load_cache(self.file_index)
            should_added_queue = len(self.data_cache_queue) < self.max_queue

        time.sleep(0.1)


class DataLoader:
    def __init__(
        self,
        files,
        dataset,
        seq_len,
        sampling_rate,
        max_sample_length=300,
        max_queue=1,
        cache_size=100,
        num_threads=1,
    ):
        self.dataset = dataset
        self.files_index = files.index
        self.files = sorted(set(copy.deepcopy(files)), key=self.files_index)
        self.file_index = 0
        self.data_cache = {}
        self.data_cache_queue = []

        self.sampling_rate = sampling_rate
        self.max_sample_length = max_sample_length
        self.cache_size = cache_size
        self.max_queue = max_queue
        self.num_threads = num_threads
        self.is_epoch_end = False
        self.seq_len = seq_len
        self.threads = []
        self.start()

    def _load_cache(self, start_idx):
        cache = {}
        self.file_index += self.cache_size
        for i in range(self.cache_size):
            idx = np.random.randint(0, len(self.files))
            file_name = self.files[idx]
            data = self.dataset[file_name]

            n_frames = data.shape[1]
            n_frames = min(self.max_sample_length * self.sampling_rate, n_frames)
            if n_frames <= self.seq_len:
                start = 0
            else:
                start = np.random.randint(0, n_frames - self.seq_len)

            spect = data[:, start : start + self.seq_len]
            if abs(np.max(spect) - np.min(spect)) < 1e-6 or not np.isfinite(spect).all():
                continue
            cache[file_name] = [
                spect,
            ]

        self.data_cache_queue.append(cache)

    def on_epoch_end(self):
        self.is_epoch_end = True
        self.join()

        self.is_epoch_end = False

        self.file_index = 0
        self.data_cache.clear()
        self.data_cache_queue.clear()

        self.start()

    def start(self):
        for _ in range(self.num_threads):
            thread = threading.Thread(target=load_data, args=(self,))
            thread.start()
            self.threads.append(thread)

    def join(self):
        for thread in self.threads:
            thread.join()
        self.threads = []

    def select_data(self):
        while len(self.data_cache) <= 0:
            if len(self.data_cache_queue) >= 1:
                self.data_cache = self.data_cache_queue.pop(0)
                break

            time.sleep(0.1)

        file_name, data = random.choice(list(self.data_cache.items()))

        del self.data_cache[file_name]
        return data

    def __len__(self):
        return len(self.files)


class DataGeneratorBatch(keras.utils.Sequence):
    def __init__(
        self,
        piano_files: list,
        mix_files: list,
        dataset_piano,
        dataset_mix,
        sampling_rate,
        batch_size=32,
        patch_length=128,
        initial_epoch=0,
        max_queue=1,
        cache_size=500,
        num_threads=1,
        epoch_max_steps=None,
    ):
        print("piano files size:{}".format(len(piano_files)))
        print("mix files size:{}".format(len(mix_files)))

        self.piano_dataloader = DataLoader(
            piano_files,
            dataset_piano,
            sampling_rate=sampling_rate,
            seq_len=patch_length,
            max_queue=max_queue,
            cache_size=cache_size,
            num_threads=num_threads,
        )

        self.mix_dataloader = DataLoader(
            mix_files,
            dataset_mix,
            sampling_rate=sampling_rate,
            seq_len=patch_length,
            max_queue=max_queue,
            cache_size=cache_size,
            num_threads=num_threads,
        )

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.patch_length = patch_length

        if epoch_max_steps is not None:
            self.batch_len = epoch_max_steps
        else:
            total_seq_length = 0
            for file in mix_files:
                length = dataset_mix[file].shape[1]
                total_seq_length += (length // self.patch_length) * self.patch_length

            self.batch_len = int((total_seq_length // self.patch_length // self.batch_size)) + 1

        # データ読み込み
        self.epoch = initial_epoch

    def on_epoch_end(self):
        self.piano_dataloader.on_epoch_end()
        self.mix_dataloader.on_epoch_end()
        self.epoch += 1

    def __getitem__(self, index):
        MIX = np.full((self.batch_size, self.patch_length, 2), 0, dtype="float32")
        PIANO = np.full((self.batch_size, self.patch_length, 2), 0, dtype="float32")

        select_num = self.batch_size
        for batch in range(select_num):
            piano_data = self.piano_dataloader.select_data()[0]
            mix_data = self.mix_dataloader.select_data()[0]

            piano_scale = np.random.uniform(0.25, 1.25)
            other_scale = np.random.uniform(0.25, 1.25)
            piano_nframes = piano_data.shape[1]
            mix_nframes = mix_data.shape[1]
            PIANO[batch, :piano_nframes, 0] = piano_data[0] * piano_scale
            PIANO[batch, :piano_nframes, 1] = piano_data[1] * piano_scale
            MIX[batch, :mix_nframes, 0] = mix_data[0] * other_scale
            MIX[batch, :mix_nframes, 1] = mix_data[1] * other_scale

        return [PIANO, MIX], [PIANO, MIX]

    def __len__(self):
        return self.batch_len
