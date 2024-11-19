from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FungiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, labels):
        """
        Arguments:
            file (string): Absolute path to the jpeg file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = []
        self.files = files
        self.labels = labels
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self._preprocess()

    def _preprocess(self):
        for file in self.files:
            image = Image.open(file)
            tensor = self.transform(image)
            self.images.append(tensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.labels[idx])
        file = self.files[idx]
        sample = {'image': image, 'label': label, 'file': file}
        return sample