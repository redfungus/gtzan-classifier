import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GTZANDataset(Dataset):
    def __init__(self, csv_file, transform=None, n_data=None):
        """
        Args:
            csv_file (string): Path to the csv file with audio paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_dataset = pd.read_csv(csv_file)
        if n_data:
            self.audio_dataset = self.audio_dataset.head(n_data)
            self.audio_dataset["genre_code"] = (
                self.audio_dataset["genre"].astype("category").cat.codes
            )
        self.transform = transform
        self.labels_count = self.audio_dataset["genre"].nunique()

    def __len__(self):
        return len(self.audio_dataset) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_path = self.audio_dataset.at[idx, "path"]
        audio_file_label = self.audio_dataset.at[idx, "genre_code"]
        audio_feature = np.load(audio_file_path)

        return torch.from_numpy(audio_feature).unsqueeze(0), audio_file_label
