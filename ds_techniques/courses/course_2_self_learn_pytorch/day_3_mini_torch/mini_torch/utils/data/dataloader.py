import numpy as np
from .dataset import Dataset

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    """
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        """
        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_num = 0
        return self

    def __next__(self):
        if self.batch_num * self.batch_size >= len(self.dataset):
            raise StopIteration

        start_idx = self.batch_num * self.batch_size
        end_idx = min((self.batch_num + 1) * self.batch_size, len(self.dataset))
        
        batch_indices = self.indices[start_idx:end_idx]
        
        samples = [self.dataset[i] for i in batch_indices]
        
        # Here, we stack the samples to create a batch.
        # This is a simple collate function.
        features, labels = zip(*samples)
        
        features_batch = np.stack(features)
        labels_batch = np.stack(labels)
        
        self.batch_num += 1
        
        # Note: We are returning numpy arrays. The user of the dataloader
        # will be responsible for converting them to mini_torch.Tensor
        return features_batch, labels_batch

    def __len__(self):
        """Returns the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
