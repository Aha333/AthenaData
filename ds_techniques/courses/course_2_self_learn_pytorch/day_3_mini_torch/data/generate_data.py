import pandas as pd
import numpy as np
import os
import shutil

def generate_base_dataframe(num_samples=100, num_features=5):
    """Generates a base pandas DataFrame for our datasets."""
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    true_weights = np.array([1.5, -2.0, 3.2, 0.8, -1.1], dtype=np.float32)
    true_bias = np.array([0.5], dtype=np.float32)
    noise = np.random.randn(num_samples) * 0.1
    y = (X @ true_weights + true_bias + noise).astype(np.float32)
    
    feature_names = [f'feature_{i+1}' for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

def generate_single_file_data(script_dir):
    """Generates a single-file Parquet dataset."""
    print("--- Generating single-file dataset ---")
    file_path = os.path.join(script_dir, "sample_data_single.parquet")
    
    if os.path.exists(file_path):
        os.remove(file_path)

    df = generate_base_dataframe()
    df.to_parquet(file_path)

    print(f"Successfully created '{file_path}'")
    print("Head of single-file data:")
    print(df.head())
    print("-" * 35 + "\n")


def generate_partitioned_data(script_dir):
    """Generates a partitioned Parquet dataset."""
    print("--- Generating partitioned dataset ---")
    dir_path = os.path.join(script_dir, "sample_data_partitioned.parquet")
    
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    df = generate_base_dataframe()
    # Add the partitioning column
    df['category'] = np.random.choice(['A', 'B'], size=len(df))
    
    # Use the 'partition_cols' argument
    df.to_parquet(dir_path, partition_cols=['category'], engine='pyarrow')

    print(f"Successfully created partitioned dataset in '{dir_path}'")
    print("Head of partitioned data (before writing):")
    print(df.head())
    print("-" * 35 + "\n")


if __name__ == "__main__":
    # Get the directory where the script is located
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    generate_single_file_data(current_script_dir)
    generate_partitioned_data(current_script_dir) 