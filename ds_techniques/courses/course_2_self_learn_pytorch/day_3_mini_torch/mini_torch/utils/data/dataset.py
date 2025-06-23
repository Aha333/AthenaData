import pandas as pd
import numpy as np

class Dataset:
    """
    An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ParquetDataset(Dataset):
    """
    Dataset for reading from a single Parquet file.
    
    Simple and straightforward implementation for single-file Parquet datasets.
    """
    def __init__(self, path, feature_columns=None, label_column=None):
        """
        Args:
            path (str): Path to the parquet file.
            feature_columns (list of str, optional): List of column names to be
                used as features. If None, all columns except the label_column are used.
            label_column (str, optional): Column name for the label. If None,
                the last column is used.
        """
        # Read the parquet file using pandas
        df = pd.read_parquet(path)
        
        if label_column is None:
            label_column = df.columns[-1]
            
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != label_column]

        self.feature_columns = feature_columns
        self.label_column = label_column
        
        # Convert to numpy arrays
        self.features = df[self.feature_columns].values.astype(np.float32)
        self.labels = df[self.label_column].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset as numpy arrays.
        """
        return self.features[idx], self.labels[idx]


# Test section
if __name__ == "__main__":
    import os
    
    print("=" * 50)
    print("Testing ParquetDataset")
    print("=" * 50)
    
    # Get the path to the data file
    # Go up two levels from this file to reach the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_path = os.path.join(project_root, "data", "sample_data_single.parquet")
    
    print(f"Looking for data at: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run the data generation script first:")
        print("python data/generate_data.py")
    else:
        # Test the dataset
        dataset = ParquetDataset(data_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Feature columns: {dataset.feature_columns}")
        print(f"Label column: {dataset.label_column}")
        print(f"Features shape: {dataset.features.shape}")
        print(f"Labels shape: {dataset.labels.shape}")
        
        # Test getting a few samples
        print(f"\nTesting __getitem__:")
        for i in range(3):
            features, label = dataset[i]
            print(f"Sample {i}: features={features}, label={label}")
        
        print("\n" + "=" * 50)
        print("Test completed successfully!") 