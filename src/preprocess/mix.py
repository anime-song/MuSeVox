import os
import librosa
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description="Mix separated audio tracks.")
parser.add_argument(
    "--input_folder",
    type=str,
    default="./separated/htdemucs",
    help="Path to the input folder containing separated tracks.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="./separated/htdemucs_mix",
    help="Path to the output folder to save mixed tracks.",
)
parser.add_argument(
    "--sampling_rate",
    type=int,
    default=44100,
    help="Sampling rate for loading audio files.",
)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
sampling_rate = args.sampling_rate
os.makedirs(output_folder, exist_ok=True)

for track_name in os.listdir(input_folder):
    track_folder = os.path.join(input_folder, track_name)
    if os.path.isdir(track_folder):
        bass_path = os.path.join(track_folder, "bass.mp3")
        drums_path = os.path.join(track_folder, "drums.mp3")
        vocals_path = os.path.join(track_folder, "vocals.mp3")

        if os.path.exists(bass_path) and os.path.exists(drums_path) and os.path.exists(vocals_path):
            bass, sr = librosa.load(bass_path, sr=sampling_rate, mono=False)
            drums, _ = librosa.load(drums_path, sr=sampling_rate, mono=False)
            vocals, _ = librosa.load(vocals_path, sr=sampling_rate, mono=False)

            # Ensure all tracks are the same length
            min_length = min(bass.shape[1], drums.shape[1], vocals.shape[1])
            bass = bass[:, :min_length]
            drums = drums[:, :min_length]
            vocals = vocals[:, :min_length]

            # Mix the tracks
            mixed = bass + drums + vocals

            # Save the mixed track
            output_path = os.path.join(output_folder, f"{track_name}_mixed.mp3")
            sf.write(output_path, mixed.T, sr, format="MP3")
