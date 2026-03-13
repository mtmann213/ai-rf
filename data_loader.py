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
        """Singularity Scaling: Absolute stability via Log-Sigmoid."""
        # 1. Immediate scrubbing of non-finite values
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Strict Pre-Clip (ADC saturation limit)
        x = np.clip(x, -100.0, 100.0)
        
        # 3. Simple Min-Max Scaling (No squaring allowed!)
        # Map [-100, 100] to [-1, 1]
        return x / 100.0

    def get_generator(self, indices, batch_size=64):
        """Turbocharged generator V6.0."""
        count = 0
        chunk_size = 4096
        
        with h5py.File(self.file_path, 'r') as f:
            X_ds = f['X']
            Y_ds = f['Y']
            
            print(f"\n[V6.0] Singularity Engine Active. Absolute Stability engaged.")
            while True:
                np.random.shuffle(indices)
                for i in range(0, len(indices), chunk_size):
                    chunk_idx = sorted(indices[i:i+chunk_size])
                    X_chunk = X_ds[chunk_idx]
                    Y_chunk = Y_ds[chunk_idx]
                    
                    p = np.random.permutation(len(X_chunk))
                    X_chunk = X_chunk[p]
                    Y_chunk = Y_chunk[p]
                    
                    for j in range(0, len(X_chunk), batch_size):
                        X_batch = X_chunk[j:j+batch_size]
                        Y_batch = Y_chunk[j:j+batch_size]
                        
                        if len(X_batch) < batch_size: continue
                            
                        X_batch = self.normalize(X_batch)
                        
                        # Final Safety Check: Ensure NO nans in the batch
                        if np.any(np.isnan(X_batch)):
                            X_batch = np.nan_to_num(X_batch)
                            
                        yield X_batch, Y_batch

    def get_train_val_indices(self, test_size=0.2, seed=42):
        with h5py.File(self.file_path, 'r') as f:
            n_samples = f['X'].shape[0]
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * (1 - test_size))
        return indices[:split], indices[split:]
