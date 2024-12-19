import numpy as np
import os

def flatten(layer_output):
    return layer_output.reshape(layer_output.shape[0], -1)

def global_avg_pooling(layer_output):
    if len(layer_output.shape) != 4:
        raise ValueError(f"Input features must be a 4D array instead of {layer_output.shape}D")
    return layer_output.mean(axis=(2, 3))

# Define the directory containing .npz files
folder_path = 'extracted_data_resnet_20'
output_path = f'{folder_path}/ed'

# Initialize a dictionary to store the aggregated sums
aggregated_sums_num = {}
aggregated_sums_denom = {}
file_count = 0

use_global_avg_pooling = True
use_flatten = False

all_data = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith('4d.npz'):
        file_path = os.path.join(folder_path, file_name)
        print(f'loading {file_path}')
        data = np.load(file_path)
        for key in data.files:
            if key not in all_data:
                all_data[key] = []
            array = data[key]
            
            if use_global_avg_pooling:
                array = global_avg_pooling(array)
            
            if use_flatten:
                array = flatten(array)
            
            all_data[key].append(array)

# Concatenate all arrays for each key
for key in all_data:
    all_data[key] = np.vstack(all_data[key])

effective_dimensionality = {}
for key, array in all_data.items():
    singular_values = np.linalg.svd(array, compute_uv=False)
    effective_dimensionality[key] = (singular_values.sum())**2 / (np.sum(singular_values**2))

output_file = os.path.join(output_path, 'new-ed.npz')
print(effective_dimensionality)
np.savez(output_file, **effective_dimensionality)