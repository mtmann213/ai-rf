import h5py
import numpy as np
import tensorflow as tf

class RadioMLDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.modulations = [
            '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
            'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
            '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
            'AM-DSB-WC', 'OOK', '16QAM'
        ]

    def normalize(self, x):
        """Event Horizon: Soft-clipping for absolute gradient stability."""
        # 1. Scrub non-finite values
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Soft-Scaling (x / (1 + |x|))
        # This maps the entire number line to (-1, 1) smoothly.
        # It is much more stable than hard clipping for backpropagation.
        return x / (1.0 + np.abs(x))

    def get_generator(self, indices, batch_size=64):
        """Turbocharged generator V7.7 (Hybrid Label Engine)."""
        chunk_size = 4096
        
        # Check if we are in the corrupted 2018 dataset (2.5M samples)
        # to decide if we need reconstruction or can trust the HDF5 labels.
        with h5py.File(self.file_path, 'r') as f:
            is_corrupted_2018 = f['X'].shape[0] > 2000000
            X_ds = f['X']
            Y_ds = f['Y'] if not is_corrupted_2018 else None
            
            mode_msg = "Reconstruction" if is_corrupted_2018 else "Native Labels"
            print(f"\n[V7.7] Event Horizon Engine Active. {mode_msg} engaged.")
            
            SAMPLES_PER_CLASS = 26 * 4096 # For reconstruction only
            
            while True:
                np.random.shuffle(indices)
                for i in range(0, len(indices), chunk_size):
                    chunk_idx = sorted(indices[i:i+chunk_size])
                    X_chunk = X_ds[chunk_idx]
                    
                    if is_corrupted_2018:
                        # Reconstruct Y for the 2018 dataset
                        Y_chunk = np.zeros((len(chunk_idx), len(self.modulations)), dtype=np.float32)
                        for idx_in_chunk, abs_idx in enumerate(chunk_idx):
                            class_idx = abs_idx // SAMPLES_PER_CLASS
                            Y_chunk[idx_in_chunk, class_idx] = 1.0
                    else:
                        # Trust the labels in the VDF file
                        Y_chunk = Y_ds[chunk_idx]
                        # Ensure Y is correctly shaped (N, 24)
                        if Y_chunk.shape[1] > 24:
                            Y_chunk = Y_chunk[:, :24]
                    
                    # Shuffle chunk
                    p = np.random.permutation(len(X_chunk))
                    X_chunk = X_chunk[p]
                    Y_chunk = Y_chunk[p]
                    
                    for j in range(0, len(X_chunk), batch_size):
                        X_batch = X_chunk[j:j+batch_size]
                        Y_batch = Y_chunk[j:j+batch_size]
                        if len(X_batch) < batch_size: continue
                        yield self.normalize(X_batch), Y_batch

    def get_train_val_indices(self, test_size=0.2, seed=42):
        with h5py.File(self.file_path, 'r') as f:
            n_samples = f['X'].shape[0]
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * (1 - test_size))
        return indices[:split], indices[split:]
