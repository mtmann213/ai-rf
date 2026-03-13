import h5py
import numpy as np
import tensorflow as tf

class RadioMLDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.modulations = [
            'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK'
        ]

    def normalize(self, x):
        """Ultra-stable normalization for extreme RF signals."""
        # 1. Scale down by the maximum value in the batch to prevent squaring overflows
        # This keeps the math within a safe floating-point range
        max_val = np.max(np.abs(x)) + 1e-8
        x_scaled = x / max_val
        
        # 2. Now calculate the L2 norm safely
        norm = np.linalg.norm(x_scaled, axis=-1, keepdims=True)
        return x_scaled / (norm + 1e-8)

    def get_generator(self, indices, batch_size=64):
        """Streams normalized data from the HDF5 file in batches."""
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_idx = sorted(indices[i:i+batch_size])
                with h5py.File(self.file_path, 'r') as f:
                    X_batch = f['X'][batch_idx]
                    Y_batch = f['Y'][batch_idx]
                
                # Apply normalization
                X_batch = self.normalize(X_batch)
                yield X_batch, Y_batch

    def get_train_val_indices(self, test_size=0.2, seed=42):
        """Returns indices for training and validation splits without loading X/Y."""
        with h5py.File(self.file_path, 'r') as f:
            n_samples = f['X'].shape[0]
        
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * (1 - test_size))
        return indices[:split], indices[split:]

if __name__ == "__main__":
    # Placeholder for local testing
    print("Opal Vanguard: Data Loader initialized. Awaiting 2018.01A dataset.")
