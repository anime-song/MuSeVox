import tensorflow as tf
import librosa
import numpy as np
import argparse
import os
from pydub import AudioSegment

from omegaconf import OmegaConf
from acoustic_token_models.acoustic_token_model import AcousticModel
from util import preprocess_wav, postprocess_wav

tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="コンフィグファイルのファイルパス")
parser.add_argument("-g_p", "--g_checkpoint_path", type=str, required=True, help="ジェネレータのモデルの重みのパス")
args = parser.parse_args()

config = OmegaConf.load(args.config)
model, intermediate_model = AcousticModel.from_pretrain(
    config=config,
    model_weight_path=args.g_checkpoint_path,
)

file = input("楽曲：")
orig_y, original_sr = librosa.load(file, sr=None, mono=False)
y = librosa.resample(orig_y, orig_sr=original_sr, target_sr=config.sampling_rate, scale=True)
y, gain_db, gain_factor = preprocess_wav(y, config.sampling_rate)
orig_y = orig_y * gain_factor

if len(y.shape) == 1:
    y = np.array([y, y])

y = y.transpose(1, 0)
y = y[np.newaxis, ...]

separated = intermediate_model(y).numpy()[0]
separated_piano = separated[0]
separated_other = separated[1]

# 元の音源とずれが発生するため、リサンプリングしてから保存
separated_piano = librosa.resample(separated_piano.T, orig_sr=config.sampling_rate, target_sr=original_sr, scale=True).T
separated_other = librosa.resample(separated_other.T, orig_sr=config.sampling_rate, target_sr=original_sr, scale=True).T

separated_piano = postprocess_wav(separated_piano, gain_db)
separated_other = postprocess_wav(separated_other, gain_db)

os.makedirs("./audio_samples", exist_ok=True)
separated_piano_audio = AudioSegment(
    (separated_piano * 32767).astype(np.int16).tobytes(),
    frame_rate=original_sr,
    sample_width=2,
    channels=separated_piano.shape[1],
)
separated_piano_audio.export("./audio_samples/separated_piano.mp3", format="mp3", bitrate="192k")

separated_other_audio = AudioSegment(
    (separated_other * 32767).astype(np.int16).tobytes(),
    frame_rate=original_sr,
    sample_width=2,
    channels=separated_other.shape[1],
)
separated_other_audio.export("./audio_samples/separated_other.mp3", format="mp3", bitrate="192k")
