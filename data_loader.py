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
        """Titanium Shield: Max-scaling for absolute numerical range control."""
        # 1. Scrub NaNs/Infs immediately
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Mean Removal (Center the signal)
        x = x - np.mean(x, axis=1, keepdims=True)
        
        # 3. Global Max Scaling: Guarantees output is strictly within [-1, 1]
        # Calculate max abs across the entire frame
        scale = np.max(np.abs(x), axis=(1, 2), keepdims=True) + 1e-8
        x = x / scale
        
        return x

    def get_generator(self, indices, batch_size=64):
        """Turbocharged generator with live stats."""
        count = 0
        chunk_size = 4096
        
        with h5py.File(self.file_path, 'r') as f:
            X_ds = f['X']
            Y_ds = f['Y']
            
            print(f"Opal Vanguard: Titanium Pipe Primed.")
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
                        
                        count += 1
                        if count % 100 == 0:
                            print(f"[Batch {count} stats: min={np.min(X_batch):.2f}, max={np.max(X_batch):.2f}]")
                        
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
