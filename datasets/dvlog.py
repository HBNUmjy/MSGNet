import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


FRAME_COUNT = 60
FEATURE_DIM = 161
MAX_VIDEOS = 360
MIN_VALID_FRAMES = 6


def _polynomial_resample(features, target_length=FRAME_COUNT, degree=3):
    source_length = features.shape[0]
    if source_length == target_length:
        return features.astype(np.float32, copy=False)
    source_positions = np.linspace(0.0, 1.0, source_length)
    target_positions = np.linspace(0.0, 1.0, target_length)
    fit_degree = min(degree, source_length - 1)
    source_basis = np.vander(source_positions, fit_degree + 1, increasing=True)
    target_basis = np.vander(target_positions, fit_degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(source_basis, features, rcond=None)[0]
    return (target_basis @ coefficients).astype(np.float32)


def preprocess_video(video):
    features = np.asarray(video, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"video features must be two-dimensional, got {features.shape}")
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"expected {FEATURE_DIM} features per frame, got {features.shape[1]}"
        )

    valid_features = features[~np.all(features == 0, axis=1)]
    if valid_features.shape[0] < MIN_VALID_FRAMES:
        return None
    if valid_features.shape[0] > FRAME_COUNT:
        return valid_features[:FRAME_COUNT].astype(np.float32, copy=False)
    return _polynomial_resample(valid_features)


class MDDD(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        split_table = pd.read_csv(self.root / "labels.csv")
        split_lookup = dict(zip(split_table["names"], split_table["split"]))

        self.features = []
        self.names = []
        self.labels = []
        self._load_class("dep_feat.pkl", 1, split, split_lookup)
        self._load_class("nondep_feat.pkl", 0, split, split_lookup)

    def _load_class(self, file_name, label, split, split_lookup):
        with (self.root / file_name).open("rb") as handle:
            data = pickle.load(handle)

        for name, user_videos in zip(data["name"], data["features"]):
            if split_lookup.get(name) != split:
                continue
            processed_videos = []
            for video in user_videos[:MAX_VIDEOS]:
                processed = preprocess_video(video)
                if processed is not None:
                    processed_videos.append(processed)
            if not processed_videos:
                continue
            self.features.append(processed_videos)
            self.names.append(name)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.features[index], self.names[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    user_features, names, labels = zip(*batch)
    lengths = torch.tensor([len(videos) for videos in user_features], dtype=torch.long)
    tensors = [
        torch.from_numpy(np.stack(videos)).float()
        for videos in user_features
    ]
    padded_features = pad_sequence(tensors, batch_first=True)
    return (padded_features, lengths, list(names)), torch.tensor(labels, dtype=torch.long)
