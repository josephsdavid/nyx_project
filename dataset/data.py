import torch
import numpy as np


class moviedataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        super().__init__()
        csv = f"{split}.csv"
        self.data = torch.from_numpy(np.genfromtxt(csv))

    def __len__(self):
        return self.data.shape[0]

    def __getitem(self, idx):
        return self.data[idx, :]


def get_dataloader(split, args):
    return torch.utils.data.DataLoader(
        moviedataset(split),
        batch_size=args["batch_size"],
        shuffle=True if split == "train" else False,
        num_workers=args["num_workers"],
    )
