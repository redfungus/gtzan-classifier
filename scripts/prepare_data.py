import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import librosa
import sys
import json
from tqdm import tqdm
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import scipy.io.wavfile
import click


def create_mel_spectogram_from_file(file_path):
    wav, sr = librosa.load(file_path, sr=None)
    return create_mel_spectogram(wav, sr)


def create_mel_spectogram(wav, sr, n_fft=2048, hop_length=1024):

    s = librosa.feature.melspectrogram(
        wav, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(s, ref=np.max)


def get_max_width_mel_spectorgram(audio_df):
    max_width = -1
    for index, row in tqdm(audio_df.iterrows(), total=audio_df.shape[0]):
        # We can also add offline dataq augmentations
        wav_file_path = row["path"]
        mel_spectogram = create_mel_spectogram_from_file(wav_file_path)
        if mel_spectogram.shape[1] > max_width:
            max_width = mel_spectogram.shape[1]
        return max_width


def get_min_max_value_mel_spectogram(audio_df):
    min_value = sys.maxsize
    max_value = -sys.maxsize
    for index, row in tqdm(audio_df.iterrows(), total=audio_df.shape[0]):
        # We can also add offline dataq augmentations
        wav_file_path = row["path"]
        mel_spectogram = create_mel_spectogram_from_file(wav_file_path)
        sample_max = np.max(mel_spectogram)
        sample_min = np.min(mel_spectogram)
        if sample_max > max_value:
            max_value = sample_max
        if sample_min < min_value:
            min_value = sample_min
        return max_value, min_value


def prepare_data_from_df(audio_df, processed_data_path):
    max_width = get_max_width_mel_spectorgram(audio_df)
    max_value, min_value = get_min_max_value_mel_spectogram(audio_df)

    normalization_values = {
        "max_width": float(max_width),
        "max_value": float(max_value),
        "min_value": float(min_value),
    }

    # Create new folders for processed data
    labels = audio_df["genre"].unique().tolist()
    for label in labels:
        Path(processed_data_path / label).mkdir(parents=True, exist_ok=True)

    audio_files = []
    for index, row in tqdm(audio_df.iterrows(), total=audio_df.shape[0]):
        # We can also add offline data augmentations
        wav_file_path = row["path"]
        mel_spectogram = create_mel_spectogram_from_file(wav_file_path)
        # Reshape all to same size
        mel_spectogram.resize(128, max_width, refcheck=False)
        # Bring values between 0 and 1
        mel_spectogram = (mel_spectogram - min_value) / (max_value - min_value)
        mel_spectogram_array_path = Path(
            processed_data_path / row["genre"] / (row["name"] + ".npy")
        )
        file_entry = {
            "name": row["name"],
            "path": mel_spectogram_array_path.resolve(),
            "genre": row["genre"],
        }

        audio_files.append(file_entry)
        with mel_spectogram_array_path.open("wb") as f:
            np.save(f, mel_spectogram)

    processed_audio_df = pd.DataFrame(audio_files)
    processed_audio_df["genre_code"] = (
        processed_audio_df["genre"].astype("category").cat.codes
    )

    return processed_audio_df, normalization_values


def augment_wav(wav, sr):

    SAMPLE_RATE = 16000

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    augmented_samples = augment(samples=wav, sample_rate=SAMPLE_RATE)

    return augmented_samples


def augment_df(audio_df, augments_per_sample=3):
    augmented_files = []
    for index, row in tqdm(audio_df.iterrows(), total=audio_df.shape[0]):
        wav_file_path = row["path"]
        wav, sr = librosa.load(wav_file_path, sr=None)
        augmented_wavs = [
            augment_wav(wav, sr) for _ in range(augments_per_sample)
        ]
        for idx, augmented_wav in enumerate(augmented_wavs):

            file_name = row["name"] + f"aug{idx}.wav"
            file_path = row["path"] + f"aug{idx}.wav"
            scipy.io.wavfile.write(
                Path(file_path), sr, augmented_wav
            )
            genre = row["genre"]

            file_entry = {
                "name": file_name,
                "path": file_path,
                "genre": genre,
                "frames": -1,
                "frame_rate": -1,
                "duration": -1,
            }
            augmented_files.append(file_entry)

    augmented_audio_df = pd.DataFrame(augmented_files)
    return augmented_audio_df


@click.command()
@click.option('--augment-data', is_flag=True, help='Number of greetings.')
@click.option('--augments-per-sample', default=3, help='Number of augmentations to create from each sample.')
def prepare_data(augment_data, augments_per_sample):
    audio_df = pd.read_csv(Path(__file__).parent.parent / ".data/data.csv")
    processed_data_path = Path(
        Path(__file__).parent.parent / "processed_data/"
    )
    processed_data_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        audio_df,
        test_size=0.2,
        random_state=1,
        stratify=audio_df[["genre"]],
    )

    if augment_data:
        augmented_train_df = augment_df(train_df, augments_per_sample)
        augmented_df = pd.concat(train_df, augmented_train_df)
        processed_augmented_df, augment_norm_values = prepare_data_from_df(
            augmented_df, processed_data_path
        )
        processed_augmented_df.to_csv(
            processed_data_path / "augmented_data.csv"
        )
        with Path(processed_data_path / "augment_normalization_values.json").open(
            "w"
        ) as json_file:
            json.dump(augment_norm_values, json_file)

    processed_train_df, train_norm_values = prepare_data_from_df(train_df, processed_data_path)
    processed_test_df, test_norm_values = prepare_data_from_df(test_df, processed_data_path)

    processed_audio_df = pd.concat(
        [processed_train_df, processed_test_df], ignore_index=True
    )

    processed_audio_df.to_csv(processed_data_path / "data.csv")
    processed_train_df.to_csv(processed_data_path / "train_data.csv")
    processed_test_df.to_csv(processed_data_path / "test_data.csv")

    with Path(processed_data_path / "train_normalization_values.json").open(
        "w"
    ) as json_file:
        json.dump(train_norm_values, json_file)

    with Path(processed_data_path / "test_normalization_values.json").open(
        "w"
    ) as json_file:
        json.dump(test_norm_values, json_file)


if __name__ == "__main__":
    prepare_data()
