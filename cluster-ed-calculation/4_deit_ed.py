import numpy as np
import os

def flatten(layer_output):
    return layer_output.reshape(layer_output.shape[0], -1)

def global_avg_pooling(layer_output):
    if len(layer_output.shape) != 4:
        raise ValueError(f"Input features must be a 4D array instead of {layer_output.shape}D")
    return layer_output.mean(axis=(2, 3))

# Define the directory containing .npz files
folder_path = 'extracted_data'
output_path = f'{folder_path}/ed'

use_global_avg_pooling = False
use_flatten = False

# Initialize a dictionary to store covariance matrices
covariance_matrices = {}
total_samples = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith('.npz'):
        file_path = os.path.join(folder_path, file_name)
        print(f'loading {file_path}')
        data = np.load(file_path)
        for key in data.files:
            array = data[key]
            array = array[:, :1024]

            if use_global_avg_pooling:
                array = global_avg_pooling(array)

            if use_flatten:
                array = flatten(array)

            if key not in covariance_matrices:
                # Initialize covariance matrix and sample count for this key
                feature_dim = array.shape[1]
                covariance_matrices[key] = np.zeros((feature_dim, feature_dim))
                total_samples[key] = 0

            # Update covariance matrix incrementally
            covariance_matrices[key] += array.T @ array
            total_samples[key] += array.shape[0]

# Compute effective dimensionality
effective_dimensionality = {}
for key, cov_matrix in covariance_matrices.items():
    # Normalize by total samples to get the final covariance matrix
    cov_matrix /= total_samples[key]

    # Compute singular values (sqrt of eigenvalues)
    singular_values = np.sqrt(np.linalg.eigvalsh(cov_matrix))
    effective_dimensionality[key] = (singular_values.sum())**2 / (np.sum(singular_values**2))

# Save results
output_file = os.path.join(output_path, 'new-ed.npz')
print(effective_dimensionality)
np.savez(output_file, **effective_dimensionality)