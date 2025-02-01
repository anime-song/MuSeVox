# MuSeVox Music Source Separation
MuSeVox is a music source separation model inspired by the concept of vocoders. The input audio signal is first converted into a spectrogram, after which features are extracted using a Transformer. The extracted features are then reconstructed into an audio signal by a Decoder, and mel spectrogram losses at multiple resolutions are computed from the generated audio. Simultaneously, this generated audio is also fed into a Discriminator, where the Discriminator loss is calculated to train the entire model.

# Dataset Preparation
## Dataset Creation
To create a dataset for piano source separation, follow these steps:

1. Source Separation (Creating Mixed Data: Bass, Vocals, Drums)
Use Demucs to perform source separation on your audio files. Run the following command to separate the audio into its components:

```bash
MuSeVox>./src/preprocess/separate.py --audio_path <audio_path>
```
Replace <audio_path> with the path to your input audio file.

2. Mixing the Separated Sources
After the sources (bass, vocals, drums) have been separated, mix them into a single audio file by running:
```bash
MuSeVox>./src/preprocess/mix.py --input_folder <folder_with_separated_sources> --output_folder <output_folder> --sampling_rate 22050
```
Replace <folder_with_separated_sources> with the directory containing the separated audio files, and <output_folder> with the destination folder where the mixed audio will be saved.

Following these steps will generate the mixed audio dataset required for training the MuSeVox model.

## Piano Audio Creation
Next, we prepare the piano audio that will be used for source separation. Please note that this repository does not support the generation or creation of piano audio. Instead, we rely on audio extracted from YouTube using the YouTube IDs provided in the [PIAST](https://hayeonbang.github.io/PIAST_dataset/) dataset. This dataset serves as the source for the piano audio used in the experiments.

## Converting Audio Files to H5 Format
To prepare the audio for training, the generated audio files need to be converted into H5 format using the provided script. This conversion is performed separately for the mixed audio and the piano audio.

**Important:**
Before proceeding, note that the mix_folder_path and piano_folder_path settings in src/configs/MuSeVox.yml refer to the locations where the H5 files will be saved, not the directories containing the original audio files. Ensure that these paths are correctly set to your desired H5 output directories.

### For Mixed Audio
Run the following command to convert the mixed audio files:

```bash
MuSeVox>./src/preprocess/convert_wav.py --audio_file_path <folder_with_mixed_audio> --sampling_rate 22050 --source mix --dataset_path <dataset_save_path>
```

- Replace <folder_with_mixed_audio> with the folder where the mixed audio files are stored.
- Replace <dataset_save_path> with the destination folder where the H5 dataset will be saved.

### For Piano Audio
Run the following command to convert the piano audio files:

```bash
MuSeVox>./src/preprocess/convert_wav.py --audio_file_path <folder_with_piano_audio> --sampling_rate 22050 --source piano --dataset_path <dataset_save_path>
```

- Replace <folder_with_piano_audio> with the folder where the piano audio files are stored.
- Replace <dataset_save_path> with the destination folder where the H5 dataset will be saved.

By following these steps, the audio data will be properly converted and formatted as H5 files, ready for training the MuSeVox model.


# Training
Once the dataset has been prepared and converted to H5 files, you can start training the MuSeVox model using the provided configuration file.

Run the following command:

```bash
MuSeVox>./src/train.py --config ./src/configs/MuSeVox.yml
```
This command loads the configuration from MuSeVox.yml and begins the training process. Make sure that your dataset and configuration settings are correctly specified before starting training.


# Inference
Once the model has been trained (pretrained model weights will be released at a later date), you can run inference using the following command:

```bash
MuSeVox>./src/predict.py --config ./src/configs/MuSeVox.yml --g_checkpoint_path <path_to_generator_weights>
```

- Replace <path_to_generator_weights> with the path to the saved Generator model weights.

After executing the script, you will be prompted to enter the path to the audio file (song) you want to process. Once the inference is complete, the results will be saved in the audio_samples folder as two separate files: separated_piano and separated_other.
