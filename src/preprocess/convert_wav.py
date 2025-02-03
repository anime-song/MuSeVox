import librosa
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import argparse
import pyloudnorm as pyln
import h5py

lock = multiprocessing.Lock()


def save_to_h5(filepath, music_name, y, source):
    with lock:  # ロックを取得
        with h5py.File(filepath, mode="a") as h5:
            group = h5.require_group(source)
            group.create_dataset(name=music_name, data=y, dtype=y.dtype, shape=y.shape)


def exists_dataset(music_name, dataset_path, source):
    filepath = os.path.join(dataset_path, "dataset") + ".hdf5"
    if not os.path.exists(filepath):
        return False

    with lock:  # ロックを取得
        with h5py.File(filepath, mode="r") as h5:
            if f"/{source}/{music_name}" in h5:
                return True
            else:
                return False


def create_dataset(files, sampling_rate, dataset_path, source):
    meter = pyln.Meter(sampling_rate)
    for train_data in files:
        try:
            f = train_data

            music_name = os.path.splitext(os.path.basename(f))[0]
            print(music_name)

            # 音声読み込み
            y, _ = librosa.load(f, sr=sampling_rate, mono=False)
            if len(y.shape) == 1:
                y = np.array([y, y])

            loudness = meter.integrated_loudness(y.T)
            y = pyln.normalize.loudness(y.T, loudness, -24.0)
            y = y.T.astype("float16")

            filepath = os.path.join(dataset_path, "dataset") + ".hdf5"

            save_to_h5(filepath, music_name, y, source=source)

        except Exception as e:
            print(music_name, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file_path", default="./Dataset/Source")
    parser.add_argument("--sampling_rate", default=22050)
    parser.add_argument("--source", choices=["piano", "mix", "instruments"])
    parser.add_argument("--dataset_path", default="./Dataset/Processed")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    files = librosa.util.find_files(args.audio_file_path, ext=["mp3", "wav", "flac"], recurse=True)

    new_files = []
    for file in files:
        music_name = os.path.splitext(os.path.basename(file))[0]
        if exists_dataset(music_name, args.dataset_path, source=args.source):
            continue
        new_files.append(file)

    n_proc = 6
    N = int(np.ceil(len(new_files) / n_proc))
    y_split = [new_files[idx : idx + N] for idx in range(0, len(new_files), N)]

    Parallel(n_jobs=n_proc, backend="multiprocessing", verbose=1)(
        [
            delayed(create_dataset)(
                files=[f],
                sampling_rate=args.sampling_rate,
                dataset_path=args.dataset_path,
                source=args.source,
            )
            for f in new_files
        ]
    )
