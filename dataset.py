import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config  # assuming config has the required transformation functions

class RetinalDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = os.listdir(self.input_dir)
        self.target_files = os.listdir(self.target_dir)

        assert len(self.input_files) == len(self.target_files), "Mismatch between input and target files count"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file = self.input_files[index]
        target_file = self.target_files[index]

        input_path = os.path.join(self.input_dir, input_file)
        target_path = os.path.join(self.target_dir, target_file)

        input_image = np.array(Image.open(input_path).convert("RGB"))  # Convert GIF to RGB
        target_image = np.array(Image.open(target_path).convert("RGB"))  # Convert TIF to RGB

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = RetinalDataset("data/train/input", "data/train/target")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
