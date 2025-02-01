import demucs.separate
import os
import argparse

parser = argparse.ArgumentParser(description="Separate audio files using Demucs")
parser.add_argument("audio_path", type=str, help="Path to the directory containing audio files")
args = parser.parse_args()

audio_path = args.audio_path

audio_files = []
for root, _, files in os.walk(audio_path):
    for file in files:
        if file.endswith(".mp3") or file.endswith(".MP3"):
            audio_files.append(os.path.join(root, file))


for file in audio_files:
    try:
        demucs.separate.main(
            [
                "--mp3",
                "-n",
                "htdemucs",
                file,
            ]
        )
    except Exception as e:
        print(e)
        continue
