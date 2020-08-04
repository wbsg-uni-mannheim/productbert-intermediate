from torchvision import datasets, transforms
from base import BaseDataLoader

from dataset.datasets import BertDataset, BertDatasetMLM
import pandas as pd


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BertDataLoader(BaseDataLoader):
    """
    DataLoader for BERT encoded sequences
    """
    def __init__(self, data_dir, batch_size, file, valid_file=None, valid_batch_size=None, shuffle=True, validation_split=-1, num_workers=1):
        
        self.data_dir = data_dir
        self.dataset = BertDataset(file)
        self.valid_batch_size = valid_batch_size
        if validation_split == -1:
            self.valid_ids = pd.read_csv(valid_file)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BertDataLoaderMLM(BaseDataLoader):
    """
    DataLoader for BERT encoded sequences
    """

    def __init__(self, data_dir, batch_size, file, valid_file=None, valid_batch_size=None, shuffle=True,
                 validation_split=-1, num_workers=1):
        self.data_dir = data_dir
        self.dataset = BertDatasetMLM(file)
        self.valid_batch_size = valid_batch_size
        if validation_split == -1:
            self.valid_ids = pd.read_csv(valid_file)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)